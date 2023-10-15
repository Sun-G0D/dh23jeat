function expandOptions() {
	var x = document.getElementById("tops-menu");
  if (x.className === "flex-hidden") {
    x.className = "flex-shown";
    console.log("show");
  } else {
    x.className = "flex-hidden";
    console.log("hide");
  }
}