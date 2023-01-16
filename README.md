# twist-optimization
Wing twist script, to make elliptical lift wing.


This is an implementation of the technique in Kevin Lane's paper from 2010.

## Usage

The current name of the base VSP model is hardcoded into the script.

When `adjust_span_stations` is `True`, your model will be written to.

# Current status

All the previously known bugs are gone. The result match our general expectations,
but have not been completely validated yet either.  

Please post on the OpenVSP users mailing list about your experiences with
this tool if you try it.

# References

Lane, Kevin & Marshall, David & McDonald, Robert. (2010). Lift Superposition and Aerodynamic Twist Optimization for Achieving Desired Lift Distributions. [10.2514/6.2010-1227](https://dx.doi.org/10.2514/6.2010-1227)
