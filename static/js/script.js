// Get the "Add Another" button and rectangle container
const addAnotherButton = document.getElementById("add-another");
const rectangleContainer = document.querySelector(".div");

// Listen for the "Add Another" button click
addAnotherButton.addEventListener("click", function() {
  // Create a new input field and append it to the container
  const newInput = document.createElement("input");
  newInput.type = "text";
  newInput.className = "rectangle";
  newInput.placeholder = "Write Your Feelings Here";
  rectangleContainer.appendChild(newInput);
});
