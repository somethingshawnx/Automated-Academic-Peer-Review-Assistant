function toggleTopicInput() {
    var checkbox = document.getElementById("deep_search");
    var box = document.getElementById("topic_box");
    box.style.display = checkbox.checked ? "block" : "none";
}

function showLoading() {
    let deepSearch = document.getElementById("deep_search").checked;
    document.getElementById("loading").style.display = "flex";
    let text = deepSearch
        ? "Deep Search enabled… downloading papers, updating index, analyzing…"
        : "Uploading paper, extracting content, running analysis…";
    document.getElementById("loading-text").innerText = text;
}
