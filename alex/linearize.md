# Linearization of the stationary streamfunction-vorticity system

## 1. Stationary problem

Let \(\Omega \subset \mathbb{R}^2\) be a two-dimensional domain with coordinates
\((x,y)\). The stationary streamfunction-vorticity formulation is

```latex
\[
\Delta \psi + \omega = 0,
\]
```

```latex
\[
\frac{1}{Re}\Delta \omega - J(\psi,\omega) - f = 0.
\]
```

Here

```latex
\[
\Delta = \partial_{xx} + \partial_{yy},
\]
```

and the Jacobian is

```latex
\[
J(\psi,\omega)
=
\psi_y \omega_x - \psi_x \omega_y.
\]
```

Equivalently, define the nonlinear operator

```latex
\[
\mathcal{F}(\psi,\omega)
=
\begin{pmatrix}
\Delta \psi + \omega \\
\frac{1}{Re}\Delta \omega - J(\psi,\omega) - f
\end{pmatrix}.
\]
```

A stationary solution \((\psi_0,\omega_0)\) satisfies

```latex
\[
\mathcal{F}(\psi_0,\omega_0)=0.
\]
```

That is,

```latex
\[
\Delta \psi_0 + \omega_0 = 0,
\]
```

```latex
\[
\frac{1}{Re}\Delta \omega_0 - J(\psi_0,\omega_0) - f = 0.
\]
```

## 2. Perturbation ansatz

We perturb the stationary solution by a small parameter \(\varepsilon\):

```latex
\[
\psi = \psi_0 + \varepsilon \phi,
\qquad
\omega = \omega_0 + \varepsilon \eta,
\]
```

where \(\phi\) and \(\eta\) are infinitesimal perturbations.

The goal is to expand

```latex
\[
\mathcal{F}(\psi_0+\varepsilon\phi,\omega_0+\varepsilon\eta)
\]
```

and keep only terms of order \(\varepsilon\).

## 3. Linearization of the first equation

Start with

```latex
\[
\Delta \psi + \omega = 0.
\]
```

Substitute the perturbation:

```latex
\[
\Delta(\psi_0+\varepsilon\phi)
+(\omega_0+\varepsilon\eta)=0.
\]
```

Using linearity of \(\Delta\),

```latex
\[
\Delta\psi_0+\omega_0
+\varepsilon(\Delta\phi+\eta)=0.
\]
```

Since \((\psi_0,\omega_0)\) is stationary,

```latex
\[
\Delta\psi_0+\omega_0=0.
\]
```

Therefore the first-order perturbation equation is

```latex
\[
\Delta\phi+\eta=0.
\]
```

## 4. Linearization of the second equation

Start with

```latex
\[
\frac{1}{Re}\Delta \omega - J(\psi,\omega) - f = 0.
\]
```

Substitute

```latex
\[
\psi = \psi_0+\varepsilon\phi,
\qquad
\omega = \omega_0+\varepsilon\eta.
\]
```

Then

```latex
\[
\frac{1}{Re}\Delta(\omega_0+\varepsilon\eta)
-J(\psi_0+\varepsilon\phi,\omega_0+\varepsilon\eta)
-f=0.
\]
```

The Laplacian term is linear:

```latex
\[
\frac{1}{Re}\Delta(\omega_0+\varepsilon\eta)
=
\frac{1}{Re}\Delta\omega_0
+\varepsilon\frac{1}{Re}\Delta\eta.
\]
```

The Jacobian \(J\) is bilinear, so

```latex
\[
J(a+b,c+d)
=
J(a,c)+J(a,d)+J(b,c)+J(b,d).
\]
```

Hence

```latex
\[
\begin{aligned}
J(\psi_0+\varepsilon\phi,\omega_0+\varepsilon\eta)
&=
J(\psi_0,\omega_0)
+\varepsilon J(\phi,\omega_0)
+\varepsilon J(\psi_0,\eta)
+\varepsilon^2 J(\phi,\eta).
\end{aligned}
\]
```

Substitute this into the second equation:

```latex
\[
\begin{aligned}
0
&=
\frac{1}{Re}\Delta\omega_0
+\varepsilon\frac{1}{Re}\Delta\eta
-J(\psi_0,\omega_0)
-\varepsilon J(\phi,\omega_0)
-\varepsilon J(\psi_0,\eta)
-\varepsilon^2J(\phi,\eta)
-f.
\end{aligned}
\]
```

Collect powers of \(\varepsilon\):

```latex
\[
\begin{aligned}
0
&=
\left(
\frac{1}{Re}\Delta\omega_0
-J(\psi_0,\omega_0)
-f
\right)
\\
&\quad
+\varepsilon
\left(
\frac{1}{Re}\Delta\eta
-J(\phi,\omega_0)
-J(\psi_0,\eta)
\right)
-\varepsilon^2J(\phi,\eta).
\end{aligned}
\]
```

The zero-order term vanishes because \((\psi_0,\omega_0)\) is stationary:

```latex
\[
\frac{1}{Re}\Delta\omega_0
-J(\psi_0,\omega_0)
-f
=0.
\]
```

Neglecting the quadratic term \(\varepsilon^2J(\phi,\eta)\), the first-order
equation is

```latex
\[
\frac{1}{Re}\Delta\eta
-J(\phi,\omega_0)
-J(\psi_0,\eta)
=0.
\]
```

## 5. Linearized system

The linearized equations for the perturbation \((\phi,\eta)\) are

```latex
\[
\Delta\phi+\eta=0,
\]
```

```latex
\[
\frac{1}{Re}\Delta\eta
-J(\phi,\omega_0)
-J(\psi_0,\eta)
=0.
\]
```

This can be written as

```latex
\[
\mathcal{L}
\begin{pmatrix}
\phi \\
\eta
\end{pmatrix}
=0,
\]
```

where \(\mathcal{L}\) is the Frechet derivative of \(\mathcal{F}\) at
\((\psi_0,\omega_0)\):

```latex
\[
\mathcal{L}
=
D\mathcal{F}(\psi_0,\omega_0).
\]
```

## 6. Block operator form

The action of \(\mathcal{L}\) is

```latex
\[
\mathcal{L}
\begin{pmatrix}
\phi \\
\eta
\end{pmatrix}
=
\begin{pmatrix}
\Delta\phi+\eta \\
\frac{1}{Re}\Delta\eta
-J(\phi,\omega_0)
-J(\psi_0,\eta)
\end{pmatrix}.
\]
```

Therefore, in block form,

```latex
\[
\mathcal{L}
=
\begin{pmatrix}
\Delta & I \\
-J(\,\cdot\,,\omega_0) & \frac{1}{Re}\Delta - J(\psi_0,\,\cdot\,)
\end{pmatrix}.
\]
```

Here

```latex
\[
J(\,\cdot\,,\omega_0)\phi = J(\phi,\omega_0),
\]
```

and

```latex
\[
J(\psi_0,\,\cdot\,)\eta = J(\psi_0,\eta).
\]
```

## 7. Expanded differential-operator form

First,

```latex
\[
J(\phi,\omega_0)
=
\phi_y(\omega_0)_x-\phi_x(\omega_0)_y.
\]
```

Therefore

```latex
\[
-J(\phi,\omega_0)
=
(\omega_0)_y\phi_x-(\omega_0)_x\phi_y.
\]
```

So the lower-left block is

```latex
\[
\mathcal{L}_{21}
=
(\omega_0)_y\partial_x-(\omega_0)_x\partial_y.
\]
```

Next,

```latex
\[
J(\psi_0,\eta)
=
(\psi_0)_y\eta_x-(\psi_0)_x\eta_y.
\]
```

Hence

```latex
\[
-J(\psi_0,\eta)
=
-(\psi_0)_y\eta_x+(\psi_0)_x\eta_y.
\]
```

So the lower-right block is

```latex
\[
\mathcal{L}_{22}
=
\frac{1}{Re}\Delta
-(\psi_0)_y\partial_x
+(\psi_0)_x\partial_y.
\]
```

Thus the full linearized operator is

```latex
\[
\boxed{
\mathcal{L}
=
\begin{pmatrix}
\Delta
&
I
\\
(\omega_0)_y\partial_x-(\omega_0)_x\partial_y
&
\frac{1}{Re}\Delta
-(\psi_0)_y\partial_x
+(\psi_0)_x\partial_y
\end{pmatrix}
}
\]
```

where

```latex
\[
\partial_x=\frac{\partial}{\partial x},
\qquad
\partial_y=\frac{\partial}{\partial y},
\qquad
\Delta=\partial_{xx}+\partial_{yy}.
\]
```

## 8. Boundary conditions for perturbations

If the stationary problem uses homogeneous streamfunction boundary conditions

```latex
\[
\psi_0\big|_{\partial\Omega}=0,
\]
```

and the perturbed solution must satisfy the same boundary condition

```latex
\[
\psi\big|_{\partial\Omega}=0,
\]
```

then

```latex
\[
\psi_0+\varepsilon\phi=0
\quad\text{on }\partial\Omega.
\]
```

Since \(\psi_0=0\) on \(\partial\Omega\), the perturbation satisfies

```latex
\[
\phi\big|_{\partial\Omega}=0.
\]
```

If wall velocities are prescribed and fixed, their perturbations are zero. In
continuous notation this means the velocity perturbation

```latex
\[
u'=\phi_y,
\qquad
v'=-\phi_x
\]
```

satisfies the homogeneous version of the velocity boundary conditions.

The vorticity perturbation \(\eta\) is not an independent boundary datum if
\(\omega\) is defined by

```latex
\[
\omega = -\Delta\psi.
\]
```

Then, at the continuous level,

```latex
\[
\eta = -\Delta\phi.
\]
```

The precise boundary treatment depends on the chosen functional setting, but the
essential point for the linearization is: all boundary data that are fixed in the
base problem become homogeneous boundary data for the perturbation.

## 9. Relation to stability

For stationary Newton correction one solves

```latex
\[
\mathcal{L}
\begin{pmatrix}
\delta\psi \\
\delta\omega
\end{pmatrix}
=
-
\mathcal{F}(\psi,\omega).
\]
```

For linear stability of the stationary solution, one studies the time-dependent
linearized perturbation equation. If the nonlinear evolution has the abstract
form

```latex
\[
\partial_t q = \mathcal{G}(q),
\qquad
q=
\begin{pmatrix}
\psi \\
\omega
\end{pmatrix},
\]
```

then near \(q_0=(\psi_0,\omega_0)^T\),

```latex
\[
\partial_t q' = D\mathcal{G}(q_0)q'.
\]
```

Instability is detected by eigenvalues \(\lambda\) with positive real part:

```latex
\[
D\mathcal{G}(q_0)q'=\lambda q',
\qquad
\operatorname{Re}\lambda>0.
\]
```

The operator \(D\mathcal{G}(q_0)\) depends on the exact time-dependent form of
the equations, while \(\mathcal{L}=D\mathcal{F}(\psi_0,\omega_0)\) above is the
Frechet derivative of the stationary residual.
