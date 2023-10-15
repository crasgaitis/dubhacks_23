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


if (document.querySelector('.loading-rectangle') && document.querySelector('#progressText')) {
  const progressBar = document.querySelector('.loading-rectangle');
  const progressText = document.getElementById('progressText');
  const completionText = document.getElementById('completionText');
  const brainProgressText = document.getElementById('brainCollect');
  const brainLotsText = document.getElementById('brainLots');
  setTimeout(() => {
    // Update text when the animation iteration is complete
    alert("bruh");
    progressText.style.display = 'none'; // Hide progress text
    completionText.style.visibility = 'visible'; // Show completion text
    brainProgressText.style.display = 'none';
    brainLotsText.style.display = 'block'; // Show completion text for "Lots of Brainwaves"
  
    setTimeout(() => {
      window.location.href = 'results-page.html';
    }, 4000);
  }, 4000);
}
