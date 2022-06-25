MPI partitions and worker groups
--------------------------------

To use MPI parallelization with simsopt, you must install the
`mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ package.

MPI simsopt programs can generally be launched using ``mpirun`` or
``mpiexec``, e.g.

.. code-block::

   mpiexec -n 32 ./name_of_your_python_driver_script

where 32 can be replaced by the number of processors you want to use.
There may be specific instructions to run MPI programs on your HPC system,
so consult the documentation for your system.
   
Given a set of :math:`M` MPI processes, an optimization problem can be
parallelized in different ways.  At one extreme, all :math:`M`
processes could work together on each evaluation of the objective
function, with the optimization algorithm itself only initiating a
single evaluation of the objective function :math:`f` at a time.  At
the other extreme, the optimization algorithm could request :math:`M`
evaluations of :math:`f` all at once, with each evaluation performed
by a single processor.  This type of parallelization can be useful for
instance when finite differences are used to evaluate gradients.  Or,
both types of parallelization could be used at the same time. To allow
all of these types of parallelization, simsopt uses the concepts of
*worker groups* and *MPI partitions*.

A *worker group* is a set of MPI processes that works together to
evaluate the objective function.  An MPI partition is a division of
the total set of :math:`M` MPI processors into one or more worker
groups.  If there are :math:`W` worker groups, each group has
approximately :math:`M/W` processors (approximate because one may need to
round up or down.)  Simsopt has a class
:obj:`simsopt.util.MpiPartition` to manage this division of
processors into worker groups.  Each MPI-aware simsopt object, such as
:obj:`simsopt.mhd.Vmec`, keeps an instance of :obj:`~simsopt.util.MpiPartition`
which can be set from its constructor.  The number of worker
groups can be set by an argument to :obj:`~simsopt.util.MpiPartition`.
For instance, to tell a Vmec object to use four worker groups, one could write

.. code-block::

   from simsopt.mhd import Vmec
   from simsopt.util import MpiPartition
   # NOTE: We don't have to specify the submodules most of the times
   
   mpi = MpiPartition(4)
   equil = Vmec("input.li383_low_res", mpi=mpi)

The same :obj:`~simsopt.util.MpiPartition` instance should be passed to the solver::

  # ... code to define an optimization problem "prob" ...
  
  from simsopt.solve import least_squares_mpi_solve
  
  least_squares_mpi_solve(prob, mpi, grad=True)

Many optimization algorithms that do not use derivatives do not
support concurrent evaluations of the objective.  In this case, the
number of worker groups, :math:`W`, should be equal to 1.  Any algorithm that
uses derivatives, such as Levenberg-Marquardt, can take advantage of
multiple worker groups to evaluate derivatives by finite
differences. If the number of parameters (i.e. independent variables)
is :math:`N`, you ideally want to set :math:`W=N+1` when using 1-sided
finite differences, and set :math:`W=2N+1` when using centered
differences.  These ideal values are not required however -
``simsopt`` will evaluate finite difference derivatives for any value
of :math:`W`, and results should be exactly independent of :math:`W`.
Other derivative-free algorithms intrinsically support
parallelization, such as HOPSPACK, though no such algorithm is
available in ``simsopt`` yet.

An MPI partition is associated with three MPI communicators, "world",
"groups", and "leaders". The "world" communicator
represents all :math:`M` MPI processors available to the program. (Normally
this communicator is the same as ``MPI_COMM_WORLD``, but it could be a
subset thereof if you wish.)  The "groups" communicator also
contains the same :math:`M` processors, but grouped into colors, with
a different color representing each worker group. Therefore
operations such as ``MPI_Send`` and ``MPI_Bcast`` on this communicator
exchange data only within one worker group.  This "groups"
communicator is therefore the one that must be used for evaluation of
the objective function.  Finally, the "leaders" communicator
only includes the :math:`W` processors that have rank 0 in the
"groups" communicator.  This communicator is used for
communicating data within a parallel optimization *algorithm* (as
opposed to within a parallelized objective function).

Given an instance of :obj:`simsopt.util.MpiPartition` named
``mpi``, the number of worker groups is available as ``mpi.ngroups``,
and the index of a given processor's group is ``mpi.group``.  The
communicators are available as ``mpi.comm_world``,
``mpi.comm_groups``, and ``mpi.comm_leaders``.  The number of
processors within the communicators can be determined from
``mpi.nprocs_world``, ``mpi.nprocs_groups``, and
``mpi.nprocs_leaders``.  The rank of the present processor within the
communicators is available as ``mpi.rank_world``, ``mpi.rank_groups``,
and ``mpi.rank_leaders``.  To determine if the present processor has
rank 0 in a communicator, you can use the variables
``mpi.proc0_world`` or ``mpi.proc0_groups``, which have type ``bool``.

