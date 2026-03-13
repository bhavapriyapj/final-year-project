async function uploadFile(){

let fileInput = document.getElementById("audioFile");

let file = fileInput.files[0];

let formData = new FormData();

formData.append("file", file);

let response = await fetch("http://127.0.0.1:5000/predict",{

method: "POST",
body: formData

});

let data = await response.json();

let resultText = "Prediction: " + data.prediction + "\n\n";

for(let key in data.probabilities){

resultText += key + ": " + (data.probabilities[key]*100).toFixed(2) + "%\n";

}

document.getElementById("result").innerText = resultText;

}
