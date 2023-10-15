function expandOptions(id) {
	var x = document.getElementById(id);
  if (x.className === "flex-hidden") {
    x.className = "flex-shown";
    console.log("show");
  } else {
    x.className = "flex-hidden";
    console.log("hide");
  }
}

function closeOptions(id){
  var y = document.getElementById(id);
  if (y.className === "flex-shown") {
    y.className = "flex-hidden";
    console.log("hide");
  }
}