node {

    // node env
    env.NODEJS_HOME = "${tool 'Node14.x'}"
    env.PATH="${env.NODEJS_HOME}/bin:${env.PATH}"
    env.NODE_OPTIONS="--max-old-space-size=8092"
    env.TRT="False"

    // Platform Vars
    def ENV = ""
    def AWS_REGION = "us-east-1"
    def GIT_CREDS_ID = ""
    def PACKAGE = "true"
    //def DOCKER_TAG = "2021-10-12"
//     def DOCKER_TAG = "py37"
    def DOCKER_TAG = "latest"


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
    // TODO: determine which test data to use
    def TEST_DATA_TIMESTAMP = "20231024" // 1.0.1 Gad branch
    def TESTS_BUCKET = "com-subtlemedical-${ENV}-build-tests"
    def PUBLIC_BUCKET = "com-subtlemedical-${ENV}-public"
    def APP_ID = ""
    def APP_NAME = ""
    def APP_VERSION = ""

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
        APP_VERSION = manifest["version"]

        for (model in manifest["defaultModels"]) {
            def zip_file = model + ".zip"
            withAWS(region: AWS_REGION){
                s3Download(file:"default_models/${zip_file}", bucket:APP_BUCKET, path:"models/${APP_ID}/${zip_file}", force:true)
            }
            //sh "mkdir default_models/${model}"
            sh "unzip -o default_models/${zip_file} -d default_models" //"${model}"
            sh "ls -l default_models/${model}/"
            sh "rm -rf default_models/${zip_file}"
            sh "rm -rf app/models"
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
            git(url: 'https://github.com/subtlemedicalinc/subtle-app-utilities.git', credentialsId: GIT_CREDS_ID, branch: "develop")
        }
        sh """
            echo "Get correct branch/version"
            cd subtle-app-utilities
            git checkout ${app_utilities_version}
            cd ..
        """

        echo "Building subtle-app-utilities dependence"
        docker.image('python:3.10-bullseye').inside(" --user 0 ") {
            sh """
                cd subtle-app-utilities/subtle_python_packages
                python3.10 setup.py bdist_wheel
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
        def data_date = "20230920"
        def file_name = "subtle_python_packages_tests_data.tar.gz"
        def download_path = "/tmp/${file_name}"
        echo 'fetching test data...'
        s3Download(
            force: true,
            file: "${download_path}",
            bucket: APP_DATA_BUCKET,
            path: "subtle_app_utilities/subtle_python_packages/${data_date}/${file_name}"
        )
        sh "tar xf ${download_path} -C ${env.WORKSPACE}/subtle-app-utilities/subtle_python_packages"
        sh "rm ${download_path}"

        def zip_file = "unit_test_data.zip"
        s3Download(file:"${zip_file}", bucket:APP_DATA_BUCKET, path:"${APP_NAME}/${TEST_DATA_TIMESTAMP}/${zip_file}", force:true)
        sh "unzip -o ${zip_file} -d app/tests"
        sh "rm ${zip_file}"

        sh '''
        if [ -d html-reports ]; then
            rm -rf html-reports
        fi
        mkdir -p html-reports
        '''

        sh 'echo "tests..."'

        docker.image("subtle/gad_py310_torch20:2023-06-01").inside("--runtime=nvidia  --user 0 --shm-size=16g"){

            sh '''
                cd $WORKSPACE/subtle-app-utilities/subtle_python_packages
                python3.10 -m pip install -r test_requirements_tf2.txt --find-links $WORKSPACE/subtle_app_utilities_bdist

                echo "starting app-utilities tests..."
                pylint --rcfile pylintrc subtle/ || true

                ls -ltr $WORKSPACE/subtle-app-utilities/subtle_python_packages/subtle/procutil/tests/data/

                # define pytest markers to include and exclude & use correct py test syntax

                #need to include subtlesynth in some of the subtle app utilities pytest

                include="build subtleapp subtlegad"
                exclude="not internal and not tf1only and not subtlesynth"
                str_markers=""

                for m in $include; do
                    if [ ${#str_markers} -gt 0 ]; then
                        str_markers="$str_markers or "
                        echo $str_markers
                    fi
                    str_markers="$str_markers $m and $exclude"
                done

                echo "using pytest markers: $str_markers"
                python3.10 -m pytest -v -m "$str_markers" \
                    --junitxml xunit-reports/xunit-result-py37.xml \
                    --html=html-reports/xunit-result-py37.html \
                    --self-contained-html

                cd $WORKSPACE
                python3.10 -m pip install --find-links subtle_app_utilities_bdist -r app/requirements.txt
                python3.10 -m pip install -r app/tests/requirements.txt

                echo "starting app unit tests..."
                python3.10 -m pytest -m "not post_build" -v app/tests/test_inference.py \
                    --junitxml xunit-reports/xunit-result-py37-pre-build.xml \
                    --html=html-reports/xunit-result-py37-pre-build.html \
                    --self-contained-html

                pylint --rcfile=pylintrc app/ || true
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
    stage("Build App Package") {
        // start building the app

        sh 'echo Building executable'
        docker.image("subtle/gad_py310_torch20:2023-06-01").inside("--runtime=nvidia  --user 0"){
            sh '''
                export PYTHON=python3.10

                rm -rf app/models
                cp -r default_models app/models

                echo "Building executable file with pyinstaller"
                ./build_app.sh
                python3.10 -m pip install pip-licenses
                pip-licenses            '''
        }
        sh '''
        if [ -d dist ]; then
            rm -rf dist
        fi
        mkdir dist
        cp -r app/dist/infer dist/infer
        cp -r build/libs dist/libs
        cp -r build/bin dist/bin/
        cp app/config.yml dist/config.yml
        cp app/run.sh dist/run.sh
        chmod +x dist/run.sh
        cp -r app/models dist/infer/models
        cp manifest.json dist/infer/manifest.json
        cp manifest.json dist/manifest.json
        git rev-parse --verify HEAD > dist/hash.txt
        python3 $WORKSPACE/subtle-app-utilities/subtle_python_packages/subtle/util/licensing.py 3000 SubtleMR 7989A8C0-A8E6-11E9-B934-238695B323F8 100 > dist/licenseMR.json
        '''
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

        // do minimal tests only for feature branches
        def tests_to_run = "post_build"
        if (env.BRANCH_NAME ==~ /(feature\/(.*)|patch\/(.*))/) {
            tests_to_run = "post_build"
        } else if (ENV != "dev") {
            tests_to_run = "not internal"
        }

//         TODO: run tests in docker.image('nvcr.io/nvidia/tensorflow:19.05-py3').inside("--runtime=nvidia") {
//         todo: to enable TRT post build test
//         TODO: if running post build tests in 19.05 --> need to build app and install tensorflow with Cuda 11.1
        docker.image("subtle/post_test_python3.10:latest").inside("--gpus all  --user 0 --shm-size=16g --env TO_TEST='${tests_to_run}' --env ENV='${ENV}'"){
            
            sh '''
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WORKSPACE/dist/infer/torch/lib/
            python3.10 -m pip install --find-links=$WORKSPACE/subtle_app_utilities_bdist -r app/requirements.txt
            
            python3.10 -m pip install -r app/tests/requirements.txt
            export POST_TEST_TAG=GPU
            python3.10 -m pytest -v -m "$TO_TEST" app/tests/test_post_build.py \
                --log-level=info \
                --junitxml xunit-reports/xunit-result-py37-post-build-gpu.xml \
                --html=html-reports/xunit-result-py37-post-build-gpu.html \
                --self-contained-html
            # use single = to run in sh (double == is used in bash)
            #if [ $ENV = "stage" ]; then
            #    python3.10 app/tests/compare_with_previous_version.py
            #fi
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
        if (fileExists("app/tests/post_build_test_data/compare_data.txt")) {
            s3Upload(file: "app/tests/post_build_test_data/compare_data.txt", bucket:"${TESTS_BUCKET}", path:"${TESTS_PATH}")
        }
        if (fileExists("app/tests/post_build_test_data/processed_data_v{APP_VERSION}.zip")) {
            s3Upload(file: "app/tests/post_build_test_data/processed_data_v{APP_VERSION}.zip", bucket:"${TESTS_BUCKET}", path:"${TESTS_PATH}")
        }

    }
    stage("Download Denoising Module") {
        echo 'fetching denoising module...'
        def zip_file = "SubtleMR_2.4.0.subtleapp"
        s3Download(file:"${zip_file}", bucket:APP_BUCKET, path:"packages/3000/${zip_file}", force:true)

        sh "unzip -o ${zip_file} -d dist/"

        sh 'cp -r $WORKSPACE/dist/licenseMR.json $WORKSPACE/dist/SubtleMR/'
    }
    
    if(PACKAGE == "false"){
        stage("Platform Package and Deploy") {
            // Remove all folders to free up space
            // TODO: allocate more space to Jenkins instance to avoid "no space left on device" error - see ticket AU-161
            sh '''
            echo "Remove all folders to free up space ..."
            rm -rf app
            rm -rf subtle-app-util*
            rm -rf subtle_app_util*
            rm -rf unit_test_data.zip
            rm -rf post_build_test_data.zip
            rm -rf default_models
            rm -rf opt_models
            docker system prune -a -f
            '''
            CONFIG_S3_PATH = "${APP_NAME}/config_files/config_${APP_VERSION}.yml"
            if (fileExists("dist/config.yml")) {
                s3Upload(file: "dist/config.yml", bucket:"${TESTS_BUCKET}", path:"${CONFIG_S3_PATH}")
            }
            MANIFEST_S3_PATH = "${APP_NAME}/config_files/manifest_${APP_VERSION}.json"
            if (fileExists("dist/manifest.json")) {
                s3Upload(file: "dist/manifest.json", bucket:"${TESTS_BUCKET}", path:"${MANIFEST_S3_PATH}")
            }

            dir('subtle-platform-utils') {
                    git(url: 'https://github.com/subtlemedicalinc/subtle-platform-utils.git', credentialsId: GIT_CREDS_ID, branch: "master")
            }
            if (ENV == "stage"){
                ENV = "staging"
            }
            sh "npm i fs-extra@8.1.0 aws-sdk archiver@4.0.2"
            sh "node ./subtle-platform-utils/buildv2.js ${ENV} ./dist"
        }

    }
}