# alpha/engine/orders.py

"""
orders.py
----------
this module contains a base order class and specific order types:
market, limit, stop, and stop-limit orders. each order can be expressed
in shares, value, or percent and can optionally be a target allocation.

this models is used by the strategy class to queue instructions for execution.
"""

from abc import ABC, abstractmethod
import pandas as pd


class Order(ABC):
    counter = 0  # a class-level counter to uniquely identify each order instance

    def __init__(self, asset, amount, created_at=None, unit='shares', trade_id=None, target=False):
        """initialize an order object.

        parameters:
        - asset: the financial instrument being traded.
        - amount: the quantity of the asset to buy/sell. Cannot be NaN.
        - created_at: timestamp of when the order was created (default: None).
        - unit: the unit in which the amount is specified ('shares', 'value', or 'percent').
        - trade_id: an optional identifier linking the order to a specific trade.
        - target: if True, the amount represents the **target** position rather than the order size.
        
        raises:
        - ValueError if the amount is NaN.
        """

        # ensure the order amount is valid
        if pd.isna(amount):
            raise ValueError(f"Amount cannot be NaN. Found in asset {asset}, created at {created_at}")

        # increment the global order counter to assign a unique ID
        Order.counter += 1

        # assign order properties
        self.asset = asset  # the asset being traded
        self.amount = amount  # the order size (in shares, value, or percent)
        self.id = Order.counter  # unique order ID
        self.created_at = created_at  # timestamp when the order was created
        self.unit = unit  # the unit of the amount ('shares', 'value', 'percent')
        self.trade_id = trade_id  # optional trade ID for tracking trades
        self.target = target  # whether the order is a target position

    def amount_in_shares(self, price, portfolio_value, current_position):
        """
        converts the order amount into the number of shares to be bought/sold.

        parameters:
        - price: The current market price of the asset.
        - portfolio_value: The total portfolio value.
        - current_position: The number of shares currently held.

        returns:
        - the number of shares to be executed based on the order's unit.
        """

        if self.unit == 'shares':
            # if the order is NOT a target, return the requested amount.
            #else if it is a target, adjust the amount to reach the target position.
            if not self.target:
                return self.amount
            else:
                return self.amount - current_position

        elif self.unit == 'value':
            # convert order value into shares by dividing by price.
            # if the order is NOT a target, return the computed shares.
            # if it is a target, adjust to reach the target position.
            if not self.target:
                return int(self.amount / price)
            else:
                return int(self.amount / price) - current_position

        elif self.unit == 'percent':
            # convert portfolio percentage into shares.
            # if the order is NOT a target, return the computed shares.
            # if it is a target, adjust to reach the target position.
            if not self.target:
                return int(portfolio_value * self.amount / price)
            else:
                return int(portfolio_value * self.amount / price) - current_position

        else:
            # raise an error for invalid unit types
            raise ValueError(f"Unknown unit {self.unit}")

    def __repr__(self):
        """returns a string representation of the order, including ID, asset, 
        and amount.
        """
        return (f"{self.__class__.__name__}<id={self.id}, asset={self.asset}, "
                f"amount={self.amount}>")

class MarketOrder(Order):
    """represents a Market Order, which executes immediately at the best 
    available price.
    """
    def __init__(self, asset, amount, created_at=None, unit='shares', trade_id=None, target=False):
        super().__init__(asset, amount, created_at, unit, trade_id, target)


class LimitOrder(Order):
    """represents a Limit Order, which executes only if the market price reaches 
    the specified limit price.
    
    Attributes:
    - limit_price: The maximum (for buy orders) or minimum (for sell orders) price at which the order can be executed.
    """
    def __init__(self, asset, amount, limit_price, created_at=None, unit='shares', trade_id=None, target=False):
        super().__init__(asset, amount, created_at, unit, trade_id, target)
        self.limit_price = limit_price  # price threshold for execution


class StopOrder(Order):
    """represents a Stop Order, which converts into a Market Order once the 
    asset reaches the stop price.
    
    Attributes:
    - stop_price: the trigger price at which the stop order turns into a market order.
    """
    def __init__(self, asset, amount, stop_price, created_at=None, unit='shares', trade_id=None, target=False):
        super().__init__(asset, amount, created_at, unit, trade_id, target)
        self.stop_price = stop_price  # trigger price for market execution


class StopLimitOrder(Order):
    """represents a Stop-Limit Order, which converts into a Limit Order once 
    the asset reaches the stop price.
    
    Attributes:
    - stop_price: The trigger price at which the order becomes active.
    - limit_price: The price limit for execution once the stop price is reached.
    """
    def __init__(self, asset, amount, limit_price, stop_price, created_at=None, unit='shares', trade_id=None, target=False):
        super().__init__(asset, amount, created_at, unit, trade_id, target)
        self.limit_price = limit_price  # price threshold for execution
        self.stop_price = stop_price  # trigger price for activation