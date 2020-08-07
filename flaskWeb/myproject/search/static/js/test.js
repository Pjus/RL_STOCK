
var apikey = "7OVAOOPNIMKJKIAX"
var symbol = "AAPL"
var requestURL = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="+ symbol +"&apikey=" + api;

var request = new XMLHttpRequest();
request.open('GET', requestURL);

request.responseType = 'json';
request.send();

request.onload = function() {
    var superHeroes = request.response;
    populateHeader(superHeroes);
    showHeroes(superHeroes);
}