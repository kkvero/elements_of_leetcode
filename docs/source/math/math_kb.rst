Knowledge Base Math
===================

.. _math-kb1-label:

Natural log and e
-----------------
| ``math.log(x)`` - returns the natural logarithm of x (to base e)
| ``math.exp(x)`` -returns e raised to the power x, where e = 2.718281… is the base of natural logarithms.
| ``math.e`` - Euler's number.

>>> math.e**3
20.085536923187664
>>> math.log(_)
3.0
# Illustrating log base e (e**a * e**b) = a + b
>>> int(math.log(math.exp(5) * math.exp(7)))  # = 5 + 7
12
>>> math.log(math.exp(5) * math.exp(7))
12.0