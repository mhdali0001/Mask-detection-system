const express = require('express');

const router = express.Router()


router.get('/', (req, res) => {

    res.render("index");
    
   });

router.get('/welcome', (req, res) => {

    res.render("welcome");
    
});

module.exports = router;