drop database if exists Alzheimer ;
create database Alzheimer ;
use Alzheimer ;

create table users (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    name VARCHAR(225),
    email VARCHAR(50), 
    password VARCHAR(50)
    );
