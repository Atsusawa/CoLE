const supplementary = document.querySelector("#supplementary-video");

for (const chapter of document.querySelectorAll(".chapter")) {
  chapter.addEventListener("click", (event) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    supplementary.currentTime = Number(chapter.dataset.time);
    supplementary.play().catch(() => {});
  });
}

const examples = {
  tai_chi: "A person is tai chi",
  shark_watercolor: "a shark is swimming in the ocean, watercolor painting",
  rainy_couple: "A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, Van Gogh style",
};
const exampleSwitch = document.querySelector(".example-switch");
const exampleButtons = exampleSwitch.querySelectorAll("button");
const examplePrompt = document.querySelector("#example-prompt");
const videos = {
  native: document.querySelector("#native-example"),
  cole: document.querySelector("#cole-example"),
};

for (const button of exampleButtons) {
  button.addEventListener("click", () => {
    if (button.getAttribute("aria-pressed") === "true") return;

    const example = button.dataset.example;
    examplePrompt.textContent = `“${examples[example]}”`;
    for (const option of exampleButtons) {
      option.setAttribute("aria-pressed", String(option === button));
    }
    for (const [method, video] of Object.entries(videos)) {
      video.pause();
      video.poster = `assets/examples/${example}/${method}.jpg`;
      video.src = `assets/examples/${example}/${method}.mp4`;
    }
  });
}

exampleSwitch.hidden = false;
