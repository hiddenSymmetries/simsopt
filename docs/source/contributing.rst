Contributing to Simsopt
=======================

We are glad you decided to contribute to ``simsopt``! During development please
follow the contribution guidlines posted here. 


Types of Contribution
^^^^^^^^^^^^^^^^^^^^^

Both big and small contributions to ``simsopt`` are welcome. Some ways you can contribute to 
``simsopt`` are:

- Submit feedback
- Report bugs
- Fix bugs
- Improve documentation
- Request new features
- Contribute new algorithms
- Implement new features

Submit Feedback
---------------

You can give your feedback on ``simsopt``  by opening an `issue <https://github.com/hiddensymmetries/simsopt/issues>`_.

If you are proposing/requesting a new feature or new algorithm:

- Explain in detail why the feature is needed with a demo problem that couldn't be implemented with existing code.
- Features that are ambitious will take time or may not be implemented at all. So, keep the scope of the feature as narrow as possible, to make it easier to implement.
- Your contributions are always welcome!


Bug Reports
-----------

To report a bug in the package, open an `issue <https://github.com/hiddensymmetries/simsopt/issues>`_.

Please include in your bug report:

* Your operating system type (mac or linux) and version.
* Pertinent details about your local setup (such as MPI and compiler info) that might be helpful in troubleshooting.
* Steps to reproduce the bug.

Fix Bugs
--------

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement New Features or Algorithms
------------------------------------

First become familiar with simsopt code (at the least, the subpackage you are contributing to).
Quickest way is to reach out to the developers and ask for help. Simsopt enforces some code quality
throught unit-tests. Have the unit tests ready to go with your code. Having few examples showcasing
a problem or two solved with the new feature would be even better.

Improve Documentation
---------------------

If you feel the documentation is lagging at any place, please feel
free to submit a PR focused on fixing or improving the 
documentation. Steps to build documentation locally can be found `here <https://github.com/hiddenSymmetries/simsopt/tree/contributing/docs>`_.


Code development Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


1. Log in to GitHub and fork ``simsopt`` repository. 
   This will create a duplicate at 'github.com/<your_account>/simsopt' 
   which is your personal fork (or just 'your fork'), as opposed to this repository
   (github.com/hiddensymmetries/simsopt), which is conventionally referred to as upstream repository in git parlance.

2. Clone your fork to your machine by opening a console and executing

   .. code-block::

        git clone https://github.com/<your_account>/simsopt.git

   Make sure to clone your fork, not the upstream repo. This will create a
   directory called ``simsopt``. Navigate to it and execute

   .. code-block::

        git remote add upstream https://github.com/hiddensymmetries/simsopt.git

   In this way, your machine will know of both your fork (which git calls
   `origin`) and the upstream repository (`upstream`).

3. During development, you make changes to the code in your fork.
   code. To prevent frequent reinstallation of simsopt after each modification, 
   and to reflect the changes immediately, install ``simsopt`` as editable.

   .. code-block::
	
        pip install -e .

4. Using a new branch to start implementing your changes would be a good idea.

   .. code-block::

        git checkout -b <your_branch_name>

5. Use git add and commit commands to save your changes.
    
   .. code-block::

        git add <your_new_file_or_modified_file>
        git commit -m "Brief message highlighting the changes implemented"

6. Make sure run_tests, run_tests_mpi, and examples/run_examples all pass. Running these locally will help you to catch bugs while developing.

7. Before submitting your changes, run ``autopep8`` to fix formatting issues using the supplied ``run_autopep`` script.
   Don't forget to run step 5, once again.

9. Push the changes to github. 

    .. code block::
        git push

10. Once the changes are in your fork you can submit a pull-request. PRs will only be merged if run_tests, 
    run_tests_mpi, and examples/run_examples all pass. Request at least 1 review from the ``simsopt`` 
    developers (Bharat Medasani, Matt Landreman, Florian Wechshung). The last reviewer to approve will be in charge of merging.
    Your contributions will be reviewed and merged to upstream repository if ``simsopt`` developers are 
    satisfied with the code quality. Please note this is not a full tutorial on using git and you need to know additional
    git commands to be more efficient or not to get stuck with git conflicts.
