const express = require("express");
const path = require('path');
const dotenv = require('dotenv');
const mysql = require("mysql");
const app = express()
const {spawn} = require('child_process');
const server = require('http').createServer();
const port = process.env.PORT || 3000;

dotenv.config({ path : './.env'});
// 
var db = mysql.createConnection({
    host: process.env.DATABASE_HOST,
     user: process.env.DATABASE_USER,
     password: process.env.DATABASE_PASSWORD,
    database: process.env.DATABASE
 });

const publicDirectory = path.join(__dirname ,'./public' );
app.use(express.static(publicDirectory));
app.use(express.urlencoded({extended:false}));
app.use(express.json());
app.set('view engine','hbs');


db.connect(function(err) {
     console.log("Mysql Connected....");
});
app.get('/python', (req, res) => {


   // var dataToSend;
    // spawn new child process to call the python script
    const pythonm = spawn('python', ['Mask detection.py'],{  windowsVerbatimArguments:true });
    // collect data from script
    pythonm.stdout.on('data', function (data) {
        console.log('Pipe data from python script ...');
    //    dataToSend = data.toString();
    });
      
    // in close event we are sure that stream from child process is closed
    pythonm.on('close', (code) => {

        // send data to browser
        res.send()  
 
    });
    
    
});
app.use('/',require('./routes/pages'));
app.use('/auth',require('./routes/auth'));
app.use('/auth',require('./routes/massege'));



app.listen(port, () => console.log(`Listening on ${port}`));