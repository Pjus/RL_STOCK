let today = new Date();   

let year = today.getFullYear(); // 년도
let month = today.getMonth() + 1;  // 월
let date = today.getDate();  // 날짜
let day = today.getDay();  // 요일


function search_stock(){
    var apikey = "d2b83a9cbe59bd13f8c7615015e41b0e";
    var ticker = document.getElementById("ticker").value;
    var requestURL = "https://financialmodelingprep.com/api/v3/historical-price-full/"+ ticker +"?apikey=" + apikey;
    var request = new XMLHttpRequest();
    request.open("GET", requestURL, true);
    request.responseType = 'json';
    request.send();

    // get data
    request.onload = function() {
 
        var dps1 = [], dps2= [];
        var stockChart = new CanvasJS.StockChart("chartContainer",{
            zoomEnabled:true,

          theme: "light2",
          options: {
            responsive: false,
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }]
            },
        },
          
          rangeChanged: function (e) {
            //update total count 
             var eventCountElement = document.getElementById("eventCount");
               eventCountElement.setAttribute("value", parseInt(eventCountElement.getAttribute("value")) + 1);
      
            // update event Trigger
               var triggerLogElement = document.getElementById("triggerLog");
               triggerLogElement.setAttribute("value", e.trigger);
                
           },

          exportEnabled: true,
          title:{
            text:"StockChart with Date-Time Axis"
          },
          subtitles: [{
            text: ticker + " Price (in USD)"
          }],
          charts: [{
            axisX: {
              crosshair: {
                enabled: true,
                snapToDataPoint: true
              }
            },
            axisY: {
              prefix: "$",
            //   maximum: high
            },
            data: [{
              type: "candlestick",
              yValueFormatString: "$#,###.##",
              dataPoints : dps1
            }]
          }],
          navigator: {
            data: [{
              dataPoints: dps2
            }],
            slider: {
              minimum: new Date(2020, 04, 01),
              maximum: new Date(year, month, day)
            }
          },


        });


        var data = request.response;
        var stock_data = data['historical']
        // document.write(stock_data);


        for (const dt in stock_data){
            // dt = key.
            var str = stock_data[dt];
            var date = stock_data[dt]['date'];
            var open = stock_data[dt]['open'];
            var high = stock_data[dt]['high'];
            var low = stock_data[dt]['low'];
            var close = stock_data[dt]['close'];
            var volume = stock_data[dt]['volume'];
            // document.write(date, '\n', open,'\n', high,'\n', low,'\n', close);
            // document.write ("<br>");

            dps1.push({
                x: new Date(date),
                y: [Number(open), Number(high), Number(low), Number(close)]
            });
            dps2.push({
                x: new Date(date),
                y: Number(close)
            });

        }
        stockChart.render();
      }
}

