# ODE
The files in this repository are supplemental to a project that explores the stability properties of Runge-Kutta and linear multi-step methods for solving ODE IVPs.

In particular there is code for the following:
- Solving ODE IVPs with Runge-Kutta methods (ODEs may be of any dimension and there is rudimentary step size variation),
- Solving ODE IVP's with linear multi-step methods (ODE's may be of any order),
- Plotting the linear stability domains of Runge-Kutta and linear mulit-step methods (some plots may seem erroneous however this can be rectified by manually setting the data limits of the plot i.e. plt.xlim() etc),
- Plotting the order stars for Runge-Kutta methods,
- Plotting the projections of the order stars for linear multi-step methods (for these you must provide all the solutions for w of the method's characteristic equation).

Further there are also two files that contain the details (i.e. coefficients and what not) of some Runge-Kutta and linear multi-step methods.

Finally, there are also files that can be run to produce all of the plots used in the aforementioned project (labeled for the chapters the plots appear in). It should be noted that those files will take quite some time to run (more than 10 minutes in some cases due to the number and resolution of the plots).

To use this code simply download all files into one directory and call any of the functions defined within. Using most functions is explained where they are defined and examples of their use can be found in the files labeled "ChapeterPlots...". However such explanations are minimal and in places incomplete. You will also need to have numpy, scipy and matplotlib installed in order to run the files.
