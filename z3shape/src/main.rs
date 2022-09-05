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

#[derive(PartialEq, Eq, Hash, Clone)]
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

#[derive(PartialEq, Eq, Hash, Clone)]
enum Z3Exp {
    DecareConst(String, Z3Type),
    Assert(Box<Z3Exp>),
    Equal(Box<Z3Exp>, Box<Z3Exp>),
    CheckSat,
    GetModel,
    Variable(String),
    Head(Box<Z3Exp>),
    Tail(Box<Z3Exp>),
    Plus(Box<Z3Exp>, Box<Z3Exp>),
    Int(i64),
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
            Z3Exp::Head(exp) => write!(f, "(head {:})", exp),
            Z3Exp::Tail(exp) => write!(f, "(tail {:})", exp),
            Z3Exp::Plus(exp1, exp2) => write!(f, "(+ {:} {:})", exp1, exp2),
            Z3Exp::Int(i) => write!(f, "{:}", i),
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

    for inout in model.graph.input.iter().chain(model.graph.output.iter()) {
        let name = inout.name.as_ref().unwrap().clone();
        decares.insert(Z3Exp::DecareConst(
            name.clone(),
            Z3Type::List(Box::new(Z3Type::Int)),
        ));

        let mut shape = Vec::new();
        if let onnx::type_proto::Value::TensorType(t) = inout.type_.clone().unwrap().value.unwrap()
        {
            for d in t.shape.dim.iter() {
                if let onnx::tensor_shape_proto::dimension::Value::DimValue(i) =
                    d.value.as_ref().unwrap()
                {
                    shape.push(i.clone());
                }
            }
        }

        let mut name_e = Z3Exp::Variable(name);
        for s in shape.iter() {
            let eq = Z3Exp::Assert(Box::new(Z3Exp::Equal(
                Box::new(Z3Exp::Head(Box::new(name_e.clone()))),
                Box::new(Z3Exp::Int(*s)),
            )));
            name_e = Z3Exp::Tail(Box::new(name_e));
            conditions.push(eq);
        }
    }

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
                // println!("{:#?}", node.attribute[0]);

                let i1 = node.input[0].clone() + "_shape";
                let i2 = node.input[0].clone() + "_shape";
                let o = node.output[0].clone() + "_shape";
                decares.insert(Z3Exp::DecareConst(
                    i1.clone(),
                    Z3Type::List(Box::new(Z3Type::Int)),
                ));
                decares.insert(Z3Exp::DecareConst(
                    i2.clone(),
                    Z3Type::List(Box::new(Z3Type::Int)),
                ));
                decares.insert(Z3Exp::DecareConst(
                    o.clone(),
                    Z3Type::List(Box::new(Z3Type::Int)),
                ));

                let mut i1exp = Z3Exp::Variable(i1);
                let mut i2exp = Z3Exp::Variable(i2);
                let mut oexp = Z3Exp::Variable(o);
                for _i in 0..axis {
                    let i1h = Z3Exp::Head(Box::new(i1exp.clone()));
                    let i2h = Z3Exp::Head(Box::new(i2exp.clone()));
                    let oh = Z3Exp::Head(Box::new(oexp.clone()));

                    let eq1 =
                        Z3Exp::Assert(Box::new(Z3Exp::Equal(Box::new(i1h.clone()), Box::new(i2h))));
                    let eq2 = Z3Exp::Assert(Box::new(Z3Exp::Equal(Box::new(i1h), Box::new(oh))));
                    // println!("{:}", eq1);
                    // println!("{:}", eq2);
                    conditions.push(eq1);
                    conditions.push(eq2);

                    i1exp = Z3Exp::Tail(Box::new(i1exp));
                    i2exp = Z3Exp::Tail(Box::new(i2exp));
                    oexp = Z3Exp::Tail(Box::new(oexp));
                }

                let eq_concat = Z3Exp::Assert(Box::new(Z3Exp::Equal(
                    Box::new(Z3Exp::Head(Box::new(oexp.clone()))),
                    Box::new(Z3Exp::Plus(
                        Box::new(Z3Exp::Head(Box::new(i1exp.clone()))),
                        Box::new(Z3Exp::Head(Box::new(i2exp.clone()))),
                    )),
                )));
                // println!("{:}", eq_concat);
                conditions.push(eq_concat);

                let eq_tail_i = Z3Exp::Assert(Box::new(Z3Exp::Equal(
                    Box::new(Z3Exp::Tail(Box::new(i1exp.clone()))),
                    Box::new(Z3Exp::Tail(Box::new(i2exp))),
                )));
                let eq_tail_o = Z3Exp::Assert(Box::new(Z3Exp::Equal(
                    Box::new(Z3Exp::Tail(Box::new(i1exp))),
                    Box::new(Z3Exp::Tail(Box::new(oexp))),
                )));
                // println!("{:}", eq_tail_i);
                // println!("{:}", eq_tail_o);
                conditions.push(eq_tail_i);
                conditions.push(eq_tail_o);
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
