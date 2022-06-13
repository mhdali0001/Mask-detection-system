const express = require('express');
const authControler = require('../controlers/massege');
const router = express.Router()

router.post('/massege', authControler.massege) 

module.exports = router;