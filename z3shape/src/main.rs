extern crate protobuf;

use protobuf::{CodedInputStream, Message};
use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

mod onnx;
use onnx::ModelProto;

#[derive(PartialEq, Eq, Hash)]
enum Z3Type {
    Int,
    List(Box<Z3Type>),
}

impl fmt::Display for Z3Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Z3Type::Int => write!(f, "Int"),
            Z3Type::List(ty) => write!(f, "(List {:})", ty),
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
enum Z3Exp {
    DecareConst(String, Z3Type),
    Assert(Box<Z3Exp>),
    Equal(Box<Z3Exp>, Box<Z3Exp>),
    CheckSat,
    GetModel,
    Variable(String),
}

impl fmt::Display for Z3Exp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Z3Exp::DecareConst(val, ty) => write!(f, "(declare-const {:} {:})", val, ty),
            Z3Exp::Assert(exp) => write!(f, "(assert {:})", exp),
            Z3Exp::Equal(exp1, exp2) => write!(f, "(= {:} {:})", exp1, exp2),
            Z3Exp::CheckSat => write!(f, "(check-sat)"),
            Z3Exp::GetModel => write!(f, "(get-model)"),
            Z3Exp::Variable(var) => write!(f, "{:}", var),
        }
    }
}

fn main() {
    assert_eq!(std::env::args().len(), 2);
    let arg1 = std::env::args().nth(1).unwrap();
    let onnx_path = Path::new(&arg1);
    let file = File::open(onnx_path).expect("fail to open file");
    let mut buffered_reader = BufReader::new(file);
    let mut cis = CodedInputStream::from_buf_read(&mut buffered_reader);

    let mut model = ModelProto::new();
    model.merge_from(&mut cis).expect("fail to merge");

    let mut decares = HashSet::new();
    let mut conditions = Vec::new();

    for node in model.graph.node.iter() {
        if let Some(op_type) = &node.op_type {
            if op_type == "Relu" || op_type == "Dropout" {
                assert_eq!(node.input.len(), 1);
                assert_eq!(node.input.len(), node.output.len());

                let i = node.input[0].clone() + "_shape";
                let o = node.output[0].clone() + "_shape";
                decares.insert(Z3Exp::DecareConst(
                    i.clone(),
                    Z3Type::List(Box::new(Z3Type::Int)),
                ));
                decares.insert(Z3Exp::DecareConst(
                    o.clone(),
                    Z3Type::List(Box::new(Z3Type::Int)),
                ));
                conditions.push(Z3Exp::Assert(Box::new(Z3Exp::Equal(
                    Box::new(Z3Exp::Variable(i)),
                    Box::new(Z3Exp::Variable(o)),
                ))));
            } else if op_type == "Concat" {
                assert_eq!(node.input.len(), 2);
                assert_eq!(node.output.len(), 1);
                assert_eq!(node.attribute.len(), 1);
                let att = &node.attribute[0];
                assert_eq!(att.name, Some(String::from("axis")));
                let axis = att.i.unwrap();
                println!("{:#?}", node.attribute[0]);
            }
        }
    }

    let smt_file = arg1 + "_shape_inference.smt";
    let mut file = File::create(smt_file.clone()).unwrap();
    let mut contents = String::from("");
    for d in decares.iter() {
        contents += &format!("{:}\n", d);
    }
    for c in conditions.iter() {
        contents += &format!("{:}\n", c);
    }
    contents += &format!("{:}\n", Z3Exp::CheckSat);
    contents += &format!("{:}\n", Z3Exp::GetModel);
    file.write_all(contents.as_bytes()).unwrap();

    let output = Command::new("z3")
        .arg("-smt2")
        .arg(smt_file)
        .output()
        .unwrap_or_else(|e| panic!("failed to execute process: {}", e));

    if output.status.success() {
        let s = String::from_utf8_lossy(&output.stdout);
        print!("{}", s);
    } else {
        let s = String::from_utf8_lossy(&output.stderr);
        print!("{}", s);
    }
}
