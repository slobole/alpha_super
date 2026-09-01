mermaid.initialize({
  startOnLoad: false,
  securityLevel: "strict",
  theme: document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "neutral",
});

document$.subscribe(async () => {
  try {
    document.querySelectorAll("pre.mermaid").forEach((codeBlock) => {
      const diagram = document.createElement("div");
      diagram.className = "mermaid";
      diagram.textContent = codeBlock.textContent;
      codeBlock.replaceWith(diagram);
    });
    await mermaid.run({ querySelector: ".mermaid" });
  } catch (error) {
    const message = error?.message || error?.str || String(error);
    console.error(`Alpha Knowledge diagram error: ${message}`);
  }
});
