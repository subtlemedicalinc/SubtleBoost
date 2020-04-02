node {

    // node env
    env.NODEJS_HOME = "${tool 'Node8.x'}"
    env.PATH="${env.NODEJS_HOME}/bin:${env.PATH}"
    env.NODE_OPTIONS="--max-old-space-size=8092"

    // Platform Vars
    def ENV = ""
    def AWS_REGION = "us-east-1"
    def GIT_CREDS_ID = ""
    def PACKAGE = "true"

    def identity = awsIdentity()
    stage("Platform Env Setup") {
        if (identity.account == "701903242341") {
            ENV = "prod"
            // AWS region is different only for the app artifact bucket in stage
            AWS_REGION = "us-west-1"
        } else if (identity.account == "574639283692") {
            ENV = "stage"
            // AWS region is different only for the app artifact bucket in stage
            AWS_REGION = "us-west-2"
        } else if (identity.account == "347052790049") {
            ENV = "dev"
            if (env.BRANCH_NAME != "develop") {
              PACKAGE = "false"
            }
        }

        GIT_CREDS_ID = env.GIT_CREDS_ID

        dir('subtle-platform-utils') {
            git(
                url: 'https://github.com/subtlemedicalinc/subtle-platform-utils.git',
                credentialsId: GIT_CREDS_ID,
                branch: "master"
            )
        }
    }

    // App env
    def manifest = ""
    // TODO: make sure those buckets exist for staging and prod env later
    def APP_BUCKET = "com-subtlemedical-${ENV}-app-artifacts"
    def APP_DATA_BUCKET = "com-subtlemedical-dev-build-data"
    def TEST_DATA_TIMESTAMP = "20200304"
    def TESTS_BUCKET = "com-subtlemedical-${ENV}-build-tests"
    def PUBLIC_BUCKET = "com-subtlemedical-${ENV}-public"
    def APP_ID = ""
    def APP_NAME = ""

    stage("Checkout") {
        checkout scm

        /* get version */
        if (env.BRANCH_NAME ==~ /(master|release\/(.*)|hotfix\/(.*))/) {
            def branch_name = env.BRANCH_NAME
            sh 'git log -n 1 ${branch_name} --pretty=format:"%H" > GIT_COMMIT'
            GIT_COMMIT = readFile('GIT_COMMIT').toString()
        }
    }

    stage("Download Models") {
        sh 'echo downloading models from ${APP_BUCKET}'
        sh "rm -rf default_models"
        sh "rm -rf opt_models"
        sh "mkdir -p default_models"
        sh "mkdir -p opt_models"

        manifest = readJSON file: 'manifest.json'
        APP_ID = manifest["appId"]
        APP_NAME = manifest["aeTitle"]

        for (model in manifest["defaultModels"]) {
            def zip_file = model + ".zip"
            withAWS(region: AWS_REGION){
                s3Download(file:"default_models/${zip_file}", bucket:APP_BUCKET, path:"models/${APP_ID}/${zip_file}", force:true)
            }
            sh "mkdir default_models/${model}"
            sh "unzip -o default_models/${zip_file} -d default_models/${model}"
            sh "ls -l default_models/${model}/"
            sh "rm -rf default_models/${zip_file}"
        }

        for (model in manifest["compatibleModels"]) {
            def zip_file = model + ".zip"
            withAWS(region: AWS_REGION){
                s3Download(file:"opt_models/${zip_file}", bucket:APP_BUCKET, path:"models/${APP_ID}/${zip_file}", force:true)
            }
            sh "mkdir opt_models/${model}"
            sh "unzip -o opt_models/${zip_file} -d opt_models/${model}"
            sh "ls -l opt_models/${model}/"
            sh "rm -rf opt_models/${zip_file}"
        }
    }

    stage("Download and Build Utilities") {
        sh 'echo Cloning subtle-app-utilities dependence'
        def app_utilities_version = manifest["app_utilities_version"]
        // remove app utilities if it exists to guarantee latest version
        sh """
        if [ -r subtle_app_utilities/ ]; then
            rm -rf subtle_app_utilities
        fi
        """
        dir('subtle-app-utilities') {
            git(url: 'https://github.com/subtlemedicalinc/subtle-app-utilities.git', credentialsId: GIT_CREDS_ID, branch: "master")
        }
        sh """
            echo "Get correct branch/version"
            cd subtle-app-utilities
            git checkout ${app_utilities_version}
            cd ..
        """

        echo "Building subtle-app-utilities dependence"
        s3Download(
            force: true,
            file: "subtle-app-utilities/subtle_python_packages/dldt-build/build/dldt-artifacts.zip",
            bucket: PUBLIC_BUCKET,
            path: "dldt/dldt-artifacts.zip"
        )
        docker.image('python:3.5-stretch').inside {
            sh """
                cd subtle-app-utilities/subtle_python_packages
                python dldt-build/install_subtle.py
                python setup.py bdist_wheel
                cd ../..
            """
        }

        sh """
        if [ -r subtle_app_utilities_bdist/ ]; then
            rm -rf subtle_app_utilities_bdist
        fi
        mkdir subtle_app_utilities_bdist
        cp subtle-app-utilities/**/dist/*.whl subtle_app_utilities_bdist/
        """
    }

    stage("Pre-Build Tests") {
        def data_date = "20200310"
        def file_name = "subtle_python_packages_tests_data.tar.gz"
        def download_path = "/tmp/${file_name}"
        echo 'fetching app utils test data...'
        s3Download(
            force: true,
            file: "${download_path}",
            bucket: APP_DATA_BUCKET,
            path: "subtle_app_utilities/subtle_python_packages/${data_date}/${file_name}"
        )
        sh "tar xf ${download_path} -C ${env.WORKSPACE}/subtle-app-utilities/subtle_python_packages"
        echo "testing app util..."
        docker.image('nvcr.io/nvidia/tensorflow:19.05-py3').inside("--runtime=nvidia") {
            sh '''
                cd $WORKSPACE/subtle-app-utilities/subtle_python_packages
                python dldt-build/install_subtle.py
                pip install -r test_requirements.txt --find-links dist/
                pylint --rcfile pylintrc subtle/ || true
                pytest -v -m "build or subtleapp or subtlegad" --junitxml xunit-reports/xunit-result-py35.xml --html=html-reports/xunit-result-py35.html --self-contained-html
            '''
        }

        sh 'echo "tests..."'
        def zip_file = "unit_test_data.zip"
        s3Download(file:"${zip_file}", bucket:APP_DATA_BUCKET, path:"${APP_NAME}/${TEST_DATA_TIMESTAMP}/${zip_file}", force:true)
        sh "unzip -o ${zip_file} -d app/tests"

        sh '''
        if [ -d html-reports ]; then
            rm -rf html-reports
        fi
        mkdir -p html-reports
        '''
        docker.image("nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04").inside("--runtime=nvidia"){
            sh '''
                apt-get update
                apt-get install -y python3 python3-pip
                pip3 install --upgrade pip
                pip install --upgrade "setuptools>=45.0.0"
                pip install --find-links subtle_app_utilities_bdist -r app/requirements.txt
                pip install -r app/tests/requirements.txt
                pip install --no-deps deepbrain
                python3 -c "import tensorflow as tf; import deepbrain; print(tf.__version__); print(tf.test.is_gpu_available()); print('tensorflow and deepbrain successfully installed');"

                python3 -m pytest -m "not post_build" app/tests/ \
                    --junitxml xunit-reports/xunit-result-py35-pre-build.xml \
                    --html=html-reports/xunit-result-py35-pre-build.html \
                    --self-contained-html

                pylint --rcfile=pylintrc app/
               '''
        }

        // upload results
        if (env.BRANCH_NAME ==~ /(master|hotfix\/(.*)|release\/(.*))/) {
            TESTS_PATH = "${APP_NAME}/${GIT_COMMIT}/verifications/"
        } else if (env.BRANCH_NAME ==~ /(develop)/) {
            TESTS_PATH = "${APP_NAME}/develop/verifications/"
        } else {
            TESTS_PATH = "${APP_NAME}/feature/verifications/"
        }
        s3Upload(file: "html-reports", bucket:"${TESTS_BUCKET}", path:"${TESTS_PATH}")
        s3Upload(
            file: "subtle-app-utilities/subtle_python_packages/html-reports",
            bucket:"${TESTS_BUCKET}",
            path:"${TESTS_PATH}subtle_python_packages"
        )
    }

    stage("Build") {
        // start building the app
        sh 'echo Building executable'
        docker.image("nvidia/cuda:9.0-cudnn7-runtime-centos7").inside("--runtime=nvidia"){
            sh '''
                yum -y update && yum -y install yum-utils nvcc wget
                yum -y groupinstall development && yum -y install https://centos7.iuscommunity.org/ius-release.rpm
                yum -y install python35u python35u-pip python35u-devel
                pip3.5 install --upgrade pip
                export PYTHON=python3.5
                export PIP=pip
                rm -rf app/models
                cp -r default_models app/models
                echo "Building executable file with pyinstaller"
                ./build_app.sh
            '''
        }
        sh """
        if [ -d dist ]; then
            rm -rf dist
        fi
        mkdir dist
        cp app/dist/infer dist/infer
        cp -r build/libs dist/libs
        cp app/config.yml dist/config.yml
        cp app/run.sh dist/run.sh
        chmod +x dist/run.sh
        cp -r app/models dist/models
        cp manifest.json dist/manifest.json
        git rev-parse --verify HEAD > dist/hash.txt
        """
    }

    stage("Post build Tests") {
        sh '''
        if [ -d html-reports ]; then
            rm -rf html-reports
        fi
        mkdir -p html-reports
        '''

        // get post build test data
        def zip_file = "post_build_test_data.zip"
        s3Download(file:"${zip_file}", bucket:APP_DATA_BUCKET, path:"${APP_NAME}/${TEST_DATA_TIMESTAMP}/${zip_file}", force:true)
        sh "unzip -o ${zip_file} -d app/tests"

        docker.image("nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04").inside("--runtime=nvidia"){
            sh '''
            apt-get update
            apt-get install -y python3 python3-pip libgtk2.0-dev
            pip3 install --upgrade pip
            pip install --find-links=subtle_app_utilities_bdist -r app/requirements.txt
            
            python3 -c "from subtle.util.licensing import generate_license; import json; from datetime import date, timedelta; j = generate_license(4000, 'SubtleGAD', 'test', date.today() + timedelta(days=2)); f = open('/home/srivathsa/test.json', 'w'); json.dump(j,f); f.close();"

            mkdir -p dist/output
            cd dist
            ./run.sh ../app/tests/post_build_test_data/NO26 output config.yml test_license.json

            python3 -c  "from glob import glob; dcm_files=glob('output/**/*.dcm', recursive=True); assert len(dcm_files) == 196, 'Invalid number of output DICOM files'; print('Post build test passed!!!');"

            rm -f test_license.json
            '''
        }
    }

    if(PACKAGE == "true") {
        stage("Platform Package and Deploy") {
            dir('subtle-platform-utils') {
                    git(url: 'https://github.com/subtlemedicalinc/subtle-platform-utils.git', credentialsId: GIT_CREDS_ID, branch: "master")
            }
            if (ENV == "stage"){
                ENV = "staging"
            }
            sh "npm i fs-extra aws-sdk archiver"
            sh "node ./subtle-platform-utils/build.js ${ENV} ./dist"
        }

    }
}
