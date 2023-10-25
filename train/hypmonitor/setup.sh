mkdir -p ~/.hypmonitor
echo "# Added by HypMonitor" | tee -a ~/.bashrc
echo "export HYPMONITOR_BASE=${PWD}" | tee -a ~/.bashrc
echo "export HYPMONITOR_SRC_PATH=${1}" | tee -a ~/.bashrc
echo "export HYPMONITOR_PORT=${2}" | tee -a ~/.bashrc
echo "export HYPMONITOR_LOGDIR=~/.hypmonitor" | tee -a ~/.bashrc
echo "export PATH=${PWD}/bin:\$PATH" | tee -a ~/.bashrc
source ~/.bashrc
echo "HypMonitor setup successfully done"
