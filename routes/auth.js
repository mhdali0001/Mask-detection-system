const express = require('express');
const authControler = require('../controlers/auth');
const router = express.Router()

router.post('/login', authControler.login) 

module.exports = router;