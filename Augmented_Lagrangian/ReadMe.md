# Math for Project 2

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>

<!-- \\[ a^2 = b^2 + c^2 \\] -->
<!-- \\[  \\] -->

## Solving for x using Augmented Lagrangian Method

\\[ L(x,\lambda,c) = min_x  x^T Q x + \frac{c}{2} \|| Ax -b \||_2^2 + \lambda^T (Ax - b)  \\]

\\[\textit{Based on Prof. Li notes on ALM} \\]

\\[ \nabla_x L(x,\lambda) =  2 Q x + c A^T A x - c A^T b +  A^T \lambda \\]

\\[ \implies x =  (2Q + cA^TA)^{-1} (CA^T b - A^T \lambda) \\]

<!-- ## Solving for Lambda using KKT

\\[ L(x,\lambda) = x^T Q x + \lambda^T (Ax - b) \\]

\\[ \nabla_x L(x,\lambda) =  2 Q x + A^T \lambda \\]

\\[ \nabla_\lambda L(x,\lambda) =  Ax - b \\]

\\[\implies x^* = A^{-1}b  \\]

\\[\implies \lambda^* = -2(A^T)^{-1}Qx  \\] -->

## Links and Resources

[Tutorial on how to use Cvxopt for Python]([https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf)

[Prof. Li notes on Augmented Lagrangian method](/Project2/augmented-Lagrangian-method.pdf)

[Wikipedia page for Augmented Lagrangian method](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method)

[Very in-depth look at the Augmented Lagrangian method](https://arxiv.org/pdf/2010.11379.pdf)

[How to write Latex in Markdown](http://flennerhag.com/2017-01-14-latex/)

## Issues

mx1 vectors (x and b) needed to be flatten to m
Fine tuning epsilon
Order of operations needed to be clear for matrix calculation
c_k need to be moved to after the lambda calculation, before was incorrect (order matters)
