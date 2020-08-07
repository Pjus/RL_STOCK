



// var apikey = "7OVAOOPNIMKJKIAX";
// var symbol = "AAPL";
// var requestURL = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="+ symbol +"&outputsize=full&apikey=" + apikey;

// var request = new XMLHttpRequest();

// request.onreadystatechange = function(){
//     if (this.readyState == 4 && this.status == 200) {
//         var myArr = JSON.parse(this.responseText);
//         var data = myArr['Time Series (Daily)'];
//         console.log(data);
//     }
// };
// request.open("GET", requestURL, true);
// request.send();
// document.getElementById("id01").innerHTML = data;



function search_stock(){
    var apikey = "7OVAOOPNIMKJKIAX";
    var ticker = document.getElementById("ticker").value;
    var requestURL = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="+ ticker +"&outputsize=full&apikey=" + apikey;
    var request = new XMLHttpRequest();
    request.open("GET", requestURL, true);
    request.responseType = 'json';
    request.send();

    // get data
    request.onload = function() {
        var data = request.response;
        var stock_data = data['Time Series (Daily)']
        for (const dt in stock_data){
            // dt = key
            var date = dt;
            var str = stock_data[dt];
            var open = str['1. open'];
            var high = str['2. high'];
            var low = str['3. low'];
            var close = str['4. close'];
            var volume = str['5. volume'];
            // document.write(date, '\n', open,'\n', high,'\n', low,'\n', close);
            // document.write ("<br>");
        }
      }
}

