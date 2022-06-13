
//
const mysql = require("mysql");
// 
var db = mysql.createConnection({
     host: process.env.DATABASE_HOST,
     user: process.env.DATABASE_USER,
     password: process.env.DATABASE_PASSWORD,
     database: process.env.DATABASE
 });
 
exports.massege = (req,res)=>{
         var name = req.body.name;
         var email = req.body.email;
         var subject = req.body.subject;
         var mass = req.body.mass;
         console.log(name,email,subject,mass)
     
         var sql = "INSERT INTO message (name, email,subject,massege) VALUES (?,?,?,?)";
         db.query(sql, [name, email,subject,mass], function (err, rows) {
         });
         res.redirect("/");
         res.end();
         
    }