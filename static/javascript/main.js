// const http = require('HTTP')
var loadFile = function(event) {
	var image = document.getElementById('output');
	image.src = URL.createObjectURL(event.target.files[0]);
};
console.log('page')

// function findBird() {
// 	console.log('called findBird function');
// }