const {spawn} = require('child_process');
const express = require('express');
const router = express.Router()

router.get('/python', (req, res) => {
    console.log("hello")

    var dataToSend;
    // spawn new child process to call the python script
    const pythonm = spawn('python', ['./Mask detection.py'],{  windowsVerbatimArguments:true });
    // collect data from script
    pythonm.stdout.on('data', function (data) {
        console.log('Pipe data from python script ...');
        dataToSend = data.toString();
    });
      
    pythonm.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });
      
    // in close event we are sure that stream from child process is closed
    pythonm.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
        // send data to browser
        res.send()  
    });
    
    
})
module.exports = router;