let today = new Date();   

let year = today.getFullYear(); // 년도
let month = today.getMonth() + 1;  // 월
let date = today.getDate();  // 날짜
let day = today.getDay();  // 요일


function enterkey(){
    if (window.event.keyCode==13){
        search_stock();
    };
}

function search_stock(){
    var newData = new Array();

    var alphaapi = "7OVAOOPNIMKJKIAX";
    var ticker = document.getElementById("ticker").value;
    var requestURL = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="+ ticker +"&apikey=" + alphaapi;
    var request = new XMLHttpRequest();

    request.open("GET", requestURL, true);
    request.responseType = 'json';
    request.send();
    // get data
    request.onload = function() {
        var data = request.response;
        var stock_data = data['Time Series (Daily)'];
        for(var day in stock_data){
            var stockInfo = new Object();
            var stockData = stock_data[day]
            stockInfo.date = day;
            stockInfo.open = stockData['1. open'];
            stockInfo.high = stockData['2. high'];
            stockInfo.low = stockData['3. low'];
            stockInfo.close = stockData['4. close'];
            
            console.log(stockInfo);
            
            newData.push(stockInfo);
        }


        am4core.ready(function() {
    
            // Themes begin
            am4core.useTheme(am4themes_animated);
            // Themes end
            
            var chart = am4core.create("chartdiv", am4charts.XYChart);
            chart.paddingRight = 20;
            
            chart.dateFormatter.inputDateFormat = "yyyy-MM-dd";
            
            var dateAxis = chart.xAxes.push(new am4charts.DateAxis());
            dateAxis.renderer.grid.template.location = 0;
            
            var valueAxis = chart.yAxes.push(new am4charts.ValueAxis());
            valueAxis.tooltip.disabled = true;
            
            var series = chart.series.push(new am4charts.CandlestickSeries());
            series.dataFields.dateX = "date";
            series.dataFields.valueY = "close";
            series.dataFields.openValueY = "open";
            series.dataFields.lowValueY = "low";
            series.dataFields.highValueY = "high";
            series.simplifiedProcessing = true;
            series.tooltipText = "Open:${openValueY.value}\nLow:${lowValueY.value}\nHigh:${highValueY.value}\nClose:${valueY.value}";
            
            chart.cursor = new am4charts.XYCursor();
            
            // a separate series for scrollbar
            var lineSeries = chart.series.push(new am4charts.LineSeries());
            lineSeries.dataFields.dateX = "date";
            lineSeries.dataFields.valueY = "close";
            
            // need to set on default state, as initially series is "show"
            lineSeries.defaultState.properties.visible = false;
            
            // hide from legend too (in case there is one)
            lineSeries.hiddenInLegend = true;
            lineSeries.fillOpacity = 0.5;
            lineSeries.strokeOpacity = 0.5;
            
            var scrollbarX = new am4charts.XYChartScrollbar();
            scrollbarX.series.push(lineSeries);
            chart.scrollbarX = scrollbarX;
        
            chart.data = newData;
            // console.log(newData);
            });    
    }
}