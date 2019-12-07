const express = require("express");
const app = express();
const fs = require('fs');
const bodyParser = require('body-parser');

const PORT = process.env.PORT || 5000

app.use(bodyParser.json({limit: '50mb'}));
app.use(bodyParser.urlencoded({extended: true}));


app.get('/',function(req,res){
  res.sendFile(__dirname+'/public/index.html');
});


app.listen(PORT);
console.log(PORT);