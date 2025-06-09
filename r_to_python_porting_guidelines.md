# R to Python Porting Guidelines

This document provides guidelines for porting R code to Python while maintaining exact equivalence in behavior, numerical results, and API. These guidelines are specifically tailored for the bartMachine port project, but can be applied to other R-to-Python porting projects as well.

## General Principles

1. **Exact Equivalence**: The Python implementation should produce the same numerical results as the R implementation, with the same behavior and API.

2. **Side-by-Side Comparison**: When porting R code to Python, always have the R code and Python code open side by side to ensure that the Python code is an exact port of the R code.

3. **Incremental Testing**: Test each function as it is ported to ensure that it produces the same results as the R implementation.

4. **Documentation**: Document any differences between the R and Python implementations, including the reasons for the differences and their impact on results.

## Naming Conventions

1. **Function Names**: Convert R function names from camelCase to snake_case, following Python naming conventions.

   ```r
   # R
   bartMachine <- function(...) { ... }
   ```

   ```python
   # Python
   def bart_machine(...):
       ...
   ```

2. **Variable Names**: Convert R variable names from camelCase to snake_case, following Python naming conventions.

   ```r
   # R
   numTrees <- 50
   ```

   ```python
   # Python
   num_trees = 50
   ```

3. **Class Names**: Keep class names in CamelCase, following Python naming conventions.

   ```r
   # R
   BartMachine <- setRefClass("BartMachine", ...)
   ```

   ```python
   # Python
   class BartMachine:
       ...
   ```

## Data Types

1. **Vectors**: R vectors should be converted to NumPy arrays or pandas Series in Python.

   ```r
   # R
   x <- c(1, 2, 3, 4, 5)
   ```

   ```python
   # Python
   import numpy as np
   x = np.array([1, 2, 3, 4, 5])
   ```

2. **Matrices**: R matrices should be converted to NumPy arrays in Python.

   ```r
   # R
   m <- matrix(1:9, nrow=3, ncol=3)
   ```

   ```python
   # Python
   import numpy as np
   m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   ```

3. **Data Frames**: R data frames should be converted to pandas DataFrames in Python.

   ```r
   # R
   df <- data.frame(x=1:5, y=6:10)
   ```

   ```python
   # Python
   import pandas as pd
   df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [6, 7, 8, 9, 10]})
   ```

4. **Lists**: R lists should be converted to Python dictionaries or lists, depending on the context.

   ```r
   # R
   l <- list(a=1, b=2, c=3)
   ```

   ```python
   # Python
   l = {'a': 1, 'b': 2, 'c': 3}
   ```

5. **Factors**: R factors should be converted to pandas Categorical objects in Python.

   ```r
   # R
   f <- factor(c("a", "b", "a", "c"))
   ```

   ```python
   # Python
   import pandas as pd
   f = pd.Categorical(["a", "b", "a", "c"])
   ```

6. **Missing Values**: R's NA should be converted to NumPy's NaN or pandas' NA in Python.

   ```r
   # R
   x <- c(1, 2, NA, 4, 5)
   ```

   ```python
   # Python
   import numpy as np
   x = np.array([1, 2, np.nan, 4, 5])
   ```

## Control Structures

1. **For Loops**: R's for loops should be converted to Python's for loops.

   ```r
   # R
   for (i in 1:10) {
     print(i)
   }
   ```

   ```python
   # Python
   for i in range(1, 11):
       print(i)
   ```

2. **While Loops**: R's while loops should be converted to Python's while loops.

   ```r
   # R
   i <- 1
   while (i <= 10) {
     print(i)
     i <- i + 1
   }
   ```

   ```python
   # Python
   i = 1
   while i <= 10:
       print(i)
       i += 1
   ```

3. **If-Else Statements**: R's if-else statements should be converted to Python's if-else statements.

   ```r
   # R
   if (x > 0) {
     print("Positive")
   } else if (x < 0) {
     print("Negative")
   } else {
     print("Zero")
   }
   ```

   ```python
   # Python
   if x > 0:
       print("Positive")
   elif x < 0:
       print("Negative")
   else:
       print("Zero")
   ```

## Functions

1. **Function Definitions**: R's function definitions should be converted to Python's function definitions.

   ```r
   # R
   myFunction <- function(x, y=10) {
     return(x + y)
   }
   ```

   ```python
   # Python
   def my_function(x, y=10):
       return x + y
   ```

2. **Default Arguments**: R's default arguments should be converted to Python's default arguments.

   ```r
   # R
   myFunction <- function(x, y=10, z=NULL) {
     if (is.null(z)) {
       z <- x + y
     }
     return(x + y + z)
   }
   ```

   ```python
   # Python
   def my_function(x, y=10, z=None):
       if z is None:
           z = x + y
       return x + y + z
   ```

3. **Variable Arguments**: R's variable arguments should be converted to Python's variable arguments.

   ```r
   # R
   myFunction <- function(...) {
     args <- list(...)
     return(sum(unlist(args)))
   }
   ```

   ```python
   # Python
   def my_function(*args):
       return sum(args)
   ```

4. **Named Arguments**: R's named arguments should be converted to Python's keyword arguments.

   ```r
   # R
   myFunction <- function(x, y=10) {
     return(x + y)
   }
   result <- myFunction(y=5, x=10)
   ```

   ```python
   # Python
   def my_function(x, y=10):
       return x + y
   result = my_function(y=5, x=10)
   ```

## Object-Oriented Programming

1. **S3 Classes**: R's S3 classes should be converted to Python classes with appropriate methods.

   ```r
   # R
   myFunction <- function(x) {
     result <- list(value=x)
     class(result) <- "MyClass"
     return(result)
   }
   
   print.MyClass <- function(x) {
     cat("MyClass object with value", x$value, "\n")
   }
   ```

   ```python
   # Python
   class MyClass:
       def __init__(self, x):
           self.value = x
       
       def __str__(self):
           return f"MyClass object with value {self.value}"
   ```

2. **S4 Classes**: R's S4 classes should be converted to Python classes with appropriate methods.

   ```r
   # R
   setClass("MyClass", slots=c(value="numeric"))
   
   setMethod("initialize", "MyClass", function(.Object, value) {
     .Object@value <- value
     return(.Object)
   })
   
   setMethod("show", "MyClass", function(object) {
     cat("MyClass object with value", object@value, "\n")
   })
   ```

   ```python
   # Python
   class MyClass:
       def __init__(self, value):
           self.value = value
       
       def __str__(self):
           return f"MyClass object with value {self.value}"
   ```

3. **Reference Classes**: R's reference classes should be converted to Python classes with appropriate methods.

   ```r
   # R
   MyClass <- setRefClass("MyClass",
     fields=list(value="numeric"),
     methods=list(
       initialize=function(value) {
         .self$value <- value
       },
       show=function() {
         cat("MyClass object with value", .self$value, "\n")
       }
     )
   )
   ```

   ```python
   # Python
   class MyClass:
       def __init__(self, value):
           self.value = value
       
       def __str__(self):
           return f"MyClass object with value {self.value}"
   ```

## Mathematical Operations

1. **Vector Operations**: R's vector operations should be converted to NumPy array operations in Python.

   ```r
   # R
   x <- c(1, 2, 3, 4, 5)
   y <- c(6, 7, 8, 9, 10)
   z <- x + y
   ```

   ```python
   # Python
   import numpy as np
   x = np.array([1, 2, 3, 4, 5])
   y = np.array([6, 7, 8, 9, 10])
   z = x + y
   ```

2. **Matrix Operations**: R's matrix operations should be converted to NumPy array operations in Python.

   ```r
   # R
   m1 <- matrix(1:9, nrow=3, ncol=3)
   m2 <- matrix(9:1, nrow=3, ncol=3)
   m3 <- m1 %*% m2
   ```

   ```python
   # Python
   import numpy as np
   m1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   m2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
   m3 = np.matmul(m1, m2)
   ```

3. **Statistical Functions**: R's statistical functions should be converted to NumPy or SciPy functions in Python.

   ```r
   # R
   mean_val <- mean(x)
   sd_val <- sd(x)
   ```

   ```python
   # Python
   import numpy as np
   mean_val = np.mean(x)
   sd_val = np.std(x, ddof=1)  # ddof=1 for sample standard deviation
   ```

## Random Number Generation

1. **Setting Seeds**: R's set.seed() should be converted to NumPy's random.seed() in Python.

   ```r
   # R
   set.seed(123)
   ```

   ```python
   # Python
   import numpy as np
   np.random.seed(123)
   ```

2. **Random Sampling**: R's random sampling functions should be converted to NumPy's random sampling functions in Python.

   ```r
   # R
   x <- rnorm(100, mean=0, sd=1)
   ```

   ```python
   # Python
   import numpy as np
   x = np.random.normal(loc=0, scale=1, size=100)
   ```

## Error Handling

1. **Try-Catch**: R's try-catch should be converted to Python's try-except.

   ```r
   # R
   tryCatch({
     result <- myFunction(x)
   }, error=function(e) {
     print(paste("Error:", e$message))
     result <- NULL
   })
   ```

   ```python
   # Python
   try:
       result = my_function(x)
   except Exception as e:
       print(f"Error: {str(e)}")
       result = None
   ```

2. **Warnings**: R's warnings should be converted to Python's warnings.

   ```r
   # R
   if (x < 0) {
     warning("x is negative")
   }
   ```

   ```python
   # Python
   import warnings
   if x < 0:
       warnings.warn("x is negative")
   ```

## Input/Output

1. **File Reading**: R's file reading functions should be converted to Python's file reading functions.

   ```r
   # R
   data <- read.csv("data.csv")
   ```

   ```python
   # Python
   import pandas as pd
   data = pd.read_csv("data.csv")
   ```

2. **File Writing**: R's file writing functions should be converted to Python's file writing functions.

   ```r
   # R
   write.csv(data, "output.csv")
   ```

   ```python
   # Python
   import pandas as pd
   data.to_csv("output.csv", index=False)
   ```

3. **Printing**: R's printing functions should be converted to Python's printing functions.

   ```r
   # R
   print(x)
   cat("Value:", x, "\n")
   ```

   ```python
   # Python
   print(x)
   print(f"Value: {x}")
   ```

## Java Interoperability

1. **rJava**: R's rJava functions should be converted to Python's Py4J functions.

   ```r
   # R
   library(rJava)
   .jinit()
   jobj <- .jnew("java.lang.String", "Hello, World!")
   ```

   ```python
   # Python
   from py4j.java_gateway import JavaGateway
   gateway = JavaGateway()
   jobj = gateway.jvm.java.lang.String("Hello, World!")
   ```

2. **Java Method Calls**: R's Java method calls should be converted to Python's Java method calls.

   ```r
   # R
   result <- .jcall(jobj, "I", "length")
   ```

   ```python
   # Python
   result = jobj.length()
   ```

3. **Java Array Conversion**: R's Java array conversion should be converted to Python's Java array conversion.

   ```r
   # R
   jarray <- .jarray(c(1, 2, 3, 4, 5))
   ```

   ```python
   # Python
   jarray = gateway.new_array(gateway.jvm.int, 5)
   for i in range(5):
       jarray[i] = i + 1
   ```

## Special Considerations for bartMachine

1. **Java Backend**: The bartMachine package uses a Java backend, so the Python implementation should use the same Java backend through Py4J.

2. **Random Number Generation**: The bartMachine package uses random number generation extensively, so the Python implementation should ensure that the random number generation is consistent with the R implementation.

3. **Data Preprocessing**: The bartMachine package performs data preprocessing before model building, so the Python implementation should ensure that the data preprocessing is consistent with the R implementation.

4. **Model Building**: The bartMachine package builds models using the Java backend, so the Python implementation should ensure that the model building is consistent with the R implementation.

5. **Prediction**: The bartMachine package makes predictions using the Java backend, so the Python implementation should ensure that the prediction is consistent with the R implementation.

6. **Visualization**: The bartMachine package provides visualization functions, so the Python implementation should ensure that the visualization is consistent with the R implementation.

## Conclusion

By following these guidelines, you can ensure that the Python implementation of the bartMachine package is functionally identical to the R implementation, with the same behavior, numerical results, and API. This will provide users with a seamless transition between the two implementations and ensure that the Python implementation can be used as a drop-in replacement for the R implementation.
