extern crate protobuf;

use protobuf::{CodedInputStream, Message};
use std::fs::File;
use std::io::BufReader;

mod onnx;
use onnx::ModelProto;

fn main() {
    let file = File::open("squeezenet1.1-7.onnx").expect("fail to open file");
    let mut buffered_reader = BufReader::new(file);
    let mut cis = CodedInputStream::from_buf_read(&mut buffered_reader);

    let mut u = ModelProto::new();
    u.merge_from(&mut cis).expect("fail to merge");

    println!("producer name: {}", u.producer_name());

    for node in u.graph.node.iter() {
        println!("name: {:?} op_type: {:?}", node.name, node.op_type);
        for i in node.input.iter() {
            println!("input: {:?}", i);
        }
        for i in node.output.iter() {
            println!("output: {:?}", i);
        }
        for a in node.attribute.iter() {
            println!("attribute name: {:?}", a.name);
        }
    }

    for value_info in u.graph.value_info.iter() {
        println!("value_info name: {:?}", value_info.name);
    }

    for input in u.graph.input.iter() {
        println!("name: {:?}", input);
    }
}
