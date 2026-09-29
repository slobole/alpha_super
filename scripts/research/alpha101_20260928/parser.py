"""Parser for the paper's expression language (PREREG section 4; research only).

Precedence, lowest to highest: ternary `c ? a : b` (right-associative) < `||` < `&&` < comparisons `<`, `>`, `==`
(non-associative) < `+`, `-` < `*`, `/` < unary minus < `^` (right-associative). Names are case-insensitive. Numbers
may be written `2.` or `.001`. `IndClass.sector` / `.industry` / `.subindustry` are group references.
"""

from __future__ import annotations

import dataclasses
import re

TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+\.\d*|\.\d+|\d+)|(?P<name>[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)?)|(?P<op>==|\|\||&&|[-+*/^<>?:,()]))"
)
GROUP_LEVEL_TUPLE = ("sector", "industry", "subindustry")


@dataclasses.dataclass(frozen=True)
class Num:
    value: float


@dataclasses.dataclass(frozen=True)
class Var:
    name: str  # lower-case


@dataclasses.dataclass(frozen=True)
class Group:
    level: str  # sector | industry | subindustry


@dataclasses.dataclass(frozen=True)
class Call:
    name: str  # lower-case
    args: tuple


@dataclasses.dataclass(frozen=True)
class Unary:
    op: str
    x: object


@dataclasses.dataclass(frozen=True)
class Binary:
    op: str
    x: object
    y: object


@dataclasses.dataclass(frozen=True)
class Ternary:
    cond: object
    a: object
    b: object


def tokenize(text_str: str) -> list[tuple[str, str]]:
    token_list: list[tuple[str, str]] = []
    pos_int = 0
    text_str = text_str.strip()
    while pos_int < len(text_str):
        match_obj = TOKEN_RE.match(text_str, pos_int)
        if match_obj is None or match_obj.end() == pos_int:
            raise SyntaxError(f"cannot tokenize at {pos_int}: {text_str[pos_int:pos_int + 20]!r}")
        pos_int = match_obj.end()
        kind_str = match_obj.lastgroup
        if kind_str is None:
            continue
        token_list.append((kind_str, match_obj.group(kind_str)))
    return token_list


class Parser:
    def __init__(self, text_str: str):
        self.token_list = tokenize(text_str)
        self.pos_int = 0

    def peek(self, offset_int: int = 0):
        idx_int = self.pos_int + offset_int
        return self.token_list[idx_int] if idx_int < len(self.token_list) else (None, None)

    def take(self, expect_str: str | None = None) -> tuple[str, str]:
        kind_str, value_str = self.peek()
        if kind_str is None:
            raise SyntaxError("unexpected end of expression")
        if expect_str is not None and value_str != expect_str:
            raise SyntaxError(f"expected {expect_str!r}, got {value_str!r} at token {self.pos_int}")
        self.pos_int += 1
        return kind_str, value_str

    def parse(self):
        node = self.parse_ternary()
        if self.peek()[0] is not None:
            raise SyntaxError(f"trailing tokens from {self.pos_int}: {self.token_list[self.pos_int:self.pos_int + 5]}")
        return node

    def parse_ternary(self):
        cond = self.parse_or()
        if self.peek()[1] == "?":
            self.take("?")
            a = self.parse_ternary()
            self.take(":")
            b = self.parse_ternary()
            return Ternary(cond, a, b)
        return cond

    def parse_or(self):
        node = self.parse_and()
        while self.peek()[1] == "||":
            self.take()
            node = Binary("||", node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_comparison()
        while self.peek()[1] == "&&":
            self.take()
            node = Binary("&&", node, self.parse_comparison())
        return node

    def parse_comparison(self):
        node = self.parse_additive()
        if self.peek()[1] in ("<", ">", "=="):
            _, op_str = self.take()
            node = Binary(op_str, node, self.parse_additive())
        return node

    def parse_additive(self):
        node = self.parse_multiplicative()
        while self.peek()[1] in ("+", "-"):
            _, op_str = self.take()
            node = Binary(op_str, node, self.parse_multiplicative())
        return node

    def parse_multiplicative(self):
        node = self.parse_unary()
        while self.peek()[1] in ("*", "/"):
            _, op_str = self.take()
            node = Binary(op_str, node, self.parse_unary())
        return node

    def parse_unary(self):
        if self.peek()[1] == "-":
            self.take()
            return Unary("-", self.parse_unary())
        if self.peek()[1] == "+":
            self.take()
            return self.parse_unary()
        return self.parse_power()

    def parse_power(self):
        base = self.parse_primary()
        if self.peek()[1] == "^":
            self.take()
            exponent = self.parse_unary()  # right-associative; `x^-y` allowed
            return Binary("^", base, exponent)
        return base

    def parse_primary(self):
        kind_str, value_str = self.take()
        if kind_str == "num":
            return Num(float(value_str))
        if kind_str == "name":
            name_str = value_str.lower()
            if name_str.startswith("indclass."):
                level_str = name_str.split(".", 1)[1]
                if level_str not in GROUP_LEVEL_TUPLE:
                    raise SyntaxError(f"unknown IndClass level {level_str!r}")
                return Group(level_str)
            if self.peek()[1] == "(":
                self.take("(")
                arg_list = []
                if self.peek()[1] != ")":
                    arg_list.append(self.parse_ternary())
                    while self.peek()[1] == ",":
                        self.take(",")
                        arg_list.append(self.parse_ternary())
                self.take(")")
                return Call(name_str, tuple(arg_list))
            return Var(name_str)
        if value_str == "(":
            node = self.parse_ternary()
            self.take(")")
            return node
        raise SyntaxError(f"unexpected token {value_str!r} at {self.pos_int - 1}")


def parse(text_str: str):
    return Parser(text_str).parse()


def node_names(node, out_set: set | None = None) -> set:
    """All variable and call names in a tree (lower-case)."""
    out_set = set() if out_set is None else out_set
    if isinstance(node, Var):
        out_set.add(node.name)
    elif isinstance(node, Call):
        out_set.add(node.name)
        for arg in node.args:
            node_names(arg, out_set)
    elif isinstance(node, Unary):
        node_names(node.x, out_set)
    elif isinstance(node, Binary):
        node_names(node.x, out_set)
        node_names(node.y, out_set)
    elif isinstance(node, Ternary):
        node_names(node.cond, out_set)
        node_names(node.a, out_set)
        node_names(node.b, out_set)
    return out_set
