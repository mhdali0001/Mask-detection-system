

 const mysql = require("mysql");

var db = mysql.createConnection({
    host: process.env.DATABASE_HOST,
     user: process.env.DATABASE_USER,
     password: process.env.DATABASE_PASSWORD,
     database: process.env.DATABASE
 });
 
exports.login = (req,res)=>{
         var email = req.body.email;
         var password = req.body.password;
         db.query("select * from users where email = ? and password = ?",[email,password],function(error,results,fields){
             if(results.length > 0){
                 res.redirect("/welcome");
                 } else{
                res.redirect("/");
             }
             res.end();
         })  
    }