# Specify CoSA path
export COSA_DIR=

# Specify Timeloop path
export TIMELOOP_DIR=
export PATH="${PATH}:${TIMELOOP_DIR}/build:$PATH"

# Specify Gurobi path and license file loc
export GUROBI_HOME=
export GRB_LICENSE_FILE=
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"


