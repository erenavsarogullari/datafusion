// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`ScalarUDFImpl`] definitions for array_compact function.
use crate::utils::make_scalar_function;
use arrow_array::{Array, ArrayRef, GenericListArray, Int32Array, ListArray};
use arrow_schema::{DataType, Field};
use arrow_schema::DataType::{FixedSizeList, LargeList, List, Null};
use datafusion_common::cast::as_list_array;
use datafusion_common::exec_err;
use datafusion_doc::Documentation;
use datafusion_expr::{ColumnarValue, Expr, ScalarUDFImpl, Signature, Volatility};
use datafusion_macros::user_doc;
use std::any::Any;
use std::sync::Arc;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use itertools::Itertools;
use datafusion_common::tree_node::TreeNodeIterator;

make_udf_expr_and_func!(
    ArrayCompact,
    array_compact,
    array,
    "returns an array of the same type as the input argument where all NULL values have been removed.",
    array_compact_udf
);

#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns an array of the same type as the input argument where all NULL values have been removed.",
    syntax_example = "array_compact(array)",
    sql_example = r#"```sql
> select array_compact([3,1,NULL,4,NULL,2]);
+-----------------------------------------+
| array_compact(List([3,1,4,2]))              |
+-----------------------------------------+
| 1                                       |
+-----------------------------------------+
```"#,
    argument(
        name = "array",
        description = "Array expression. Can be a constant, column, or function, and any combination of array operators."
    )
)]
#[derive(Debug)]
pub struct ArrayCompact {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for ArrayCompact {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayCompact {
    pub fn new() -> Self {
        Self {
            signature: Signature::array(Volatility::Immutable),
            aliases: vec!["list_compact".to_string()],
        }
    }
}

impl ScalarUDFImpl for ArrayCompact {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "array_compact"
    }

    fn display_name(&self, args: &[Expr]) -> datafusion_common::Result<String> {
        let args_name = args.iter().map(ToString::to_string).collect::<Vec<_>>();
        if args_name.len() != 1 {
            return exec_err!("expects 1 arg, got {}", args_name.len());
        }

        Ok(format!("{}", args_name[0]))
    }

    fn schema_name(&self, args: &[Expr]) -> datafusion_common::Result<String> {
        let args_name = args
            .iter()
            .map(|e| e.schema_name().to_string())
            .collect::<Vec<_>>();
        if args_name.len() != 1 {
            return exec_err!("expects 1 arg, got {}", args_name.len());
        }

        Ok(format!("{}", args_name[0]))
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> datafusion_common::Result<DataType> {
        match &arg_types[0] {
            List(field) | FixedSizeList(field, _) => Ok(List(Arc::new(
                Field::new_list_field(field.data_type().clone(), true),
            ))),
            LargeList(field) => Ok(LargeList(Arc::new(Field::new_list_field(
                field.data_type().clone(),
                true,
            )))),
            _ => exec_err!(
                "Not reachable, data_type should be List, LargeList or FixedSizeList"
            ),
        }
    }

    fn invoke_batch(
        &self,
        args: &[ColumnarValue],
        _number_rows: usize,
    ) -> datafusion_common::Result<ColumnarValue> {
        make_scalar_function(array_compact_inner)(args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// array_compact SQL function
///
/// There is one argument for array_compact as the array.
/// `array_compact(array)`
///
/// For example:
/// > array_compact(\[3, NULL, 1, NULL, 2]) -> 3,1,2
pub fn array_compact_inner(args: &[ArrayRef]) -> datafusion_common::Result<ArrayRef> {
    if args.len() != 1 {
        return exec_err!("array_compact needs one argument");
    }

    match &args[0].data_type() {
        List(_) | LargeList(_) | FixedSizeList(_, _) => array_compact_internal(&args),
        _ => exec_err!("array_compact does not support type: {:?}", args[0].data_type()),
    }
}

fn array_compact_internal(args: &[ArrayRef]) -> datafusion_common::Result<ArrayRef> {
    let list_array = as_list_array(&args[0])?;
    println!("list_array: {:?}", list_array);
    println!("list_array Results => {:?}", list_array.iter().filter(|x| {
        println!("x: {:?}", x.iter().filter(|x1| {
            let dt = x1.as_ref().data_type();
            println!("dt: {:?}", dt);
            dt.is_null()
        }).collect_vec());
        let t = x.as_ref().unwrap().is_null(1);//data_type();
        println!("t Results => {:?}", t);
        !t
    }).collect_vec());



    // let res = Int32Array::new(list_array).iter().collect_vec();//.filter(|x| !x.as_ref().unwrap().is_empty()).collect_vec();
    // println!("non-null_array Results => {:?}", res);
    // println!("null_array Results => {:?}", list_array.as_primitive::<_>().iter().filter(|x: &Option<_>| !x.as_ref().unwrap().is_nullable())).collect_vec();;
    // let results  = list_array.nuas_list().iter().filter(|x|
    //     !x.as_ref().unwrap().is_empty()
    // ).collect_vec();
    // println!("Filtered Results => {:?}", results);
    // let temp_arr = results.;
    // Ok(temp_arr)
    // let arr = as_list_array(results);
    // let v = ListArray::from_iter_primitive::<Int32Type, _, _>(results);
    Ok(Arc::new(Int32Array::from(vec![5,5,5])))//

    // let sorted_array = array_sort_inner(args)?;
    // let result_array = as_list_array(&sorted_array)?.value(0);
    // if result_array.is_empty() {
    //     return exec_err!("array_min needs one argument as non-empty array");
    // }
    // let min_result = result_array.slice(0, 1);
    // Ok(min_result)
}
