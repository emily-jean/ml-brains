# Makefile for Spark Foreground/Background Classification Brains project.

# Customize these paths for your environment.
# -----------------------------------------------------------
spark.root=/usr/local/Cellar/apache-spark/2.3.0
hadoop.root=/usr/local/hadoop-2.7.5

app.name=Project
jar.name=MRProject-1.0-SNAPSHOT.jar
maven.jar.name=MRProject-1.0-SNAPSHOT.jar
jar.path=target/${jar.name}
local.master=local[4]

# Preproc and Classification job id
aux_job_id=1

# Training Parameters
max_depth=15
max_bins=256
num_trees=50
modifier=a

# total provisions 1 master + aws.num.workers
aws.num.workers=9
aws.instance.type=m4.large
aws.cores.per.machine=4
aws.partition.size=$(shell echo $$(( $(aws.cores.per.machine) * $(aws.num.workers) )))
#$(info Sum with Double-$$: ${aws.partition.size})

# General names
class.name=USE-MAKE-CLOUD-XXXXXXXX
model.name=m-${max_depth}-${max_bins}-${num_trees}-${modifier}

# job name
job.name=Training-${model.name}

raw_input_path=raw-data
preproc_out_path=pre-proc-data
training_path=pre-proc-data/train
testing_path=pre-proc-data/test
validation_path=pre-proc-data/validation
output_path=class-output
model_in_path=${model.name}/mdl
model_out_path=${model.name}/mdl
metrics_file_path=${model.name}/scores
python_file_path=preproc.py

# Likely no need to edit anything under this line
# ----------------------------------------------------------------------------

# - - - - - - - - - - - - - - - -
# Local Parameters

local.python.dir=src/main/resources/
local.python.preproc=${local.python.dir}/preproc.py
local.log=logs
local.conf.dir=conf
local.conf.file=${local.conf.dir}/configuration.yml
local.bootstrap.file=${local.conf.dir}/setuppython.sh

# Local Arguments to the functions
local.raw_input_path=data/${raw_input_path}
local.preproc_out_path=data/${preproc_out_path}
local.training_path=data/${training_path}
local.testing_path=data/${testing_path}
local.validation_path=data/${validation_path}
local.output_path=data/${output_path}
local.model_in_path=data/models/${model_in_path}
local.model_out_path=data/models/${model_out_path}
local.metrics_file_path=data/models/${metrics_file_path}
local.max_depth=${max_depth}
local.max_bins=${max_bins}
local.num_trees=${num_trees}
local.python_file_path=src/main/resources/${python_file_path}

# - - - - - - - - - - - - - - - -
# AWS Parameters

# AWS EMR Execution
aws.release.label=emr-5.13.0
aws.region=us-east-1

# CHANGE ME: Adjust bucket name
aws.bucket.name=dutile-neu
aws.subnet.id=subnet-2f731e72

# CHANGE ME: change to the project folder in your bucket
# for compatibility with other scripts
aws.proj.dir=ls-pdp/proj

# AWS Arguments to the functions
aws.raw_input_path=${aws.proj.dir}/data/${raw_input_path}
aws.preproc_out_path=${aws.proj.dir}/data/${preproc_out_path}
aws.training_path=${aws.proj.dir}/data/${training_path}
aws.testing_path=${aws.proj.dir}/data/${testing_path}
aws.validation_path=${aws.proj.dir}/data/${validation_path}
aws.output_path=${aws.proj.dir}/data/${output_path}
aws.model_in_path=${aws.proj.dir}/data/models/${model_in_path}
aws.model_out_path=${aws.proj.dir}/data/models/${model_out_path}
aws.metrics_file_path=${aws.proj.dir}/data/models/${metrics_file_path}
aws.max_depth=${max_depth}
aws.max_bins=${max_bins}
aws.num_trees=${num_trees}
aws.python_file_path=${aws.proj.dir}/scripts/${python_file_path}

# Other variables for makefile
aws.python.dir=${aws.proj.dir}/scripts
aws.log.dir=${aws.proj.dir}/log
aws.conf.dir=${aws.proj.dir}/conf
aws.conf.file=${aws.conf.dir}/configuration.yml
aws.bootstrap.file=${aws.conf.dir}/setup-python.sh

# -----------------------------------------------------------

# Compiles code and builds jar (with dependencies).
jar:
	mvn clean package
	cp target/${maven.jar.name} ${jar.name}

# Removes local.preproc.out.path directory.
clean-local.preproc.out.path:
	rm -rf ${local.preproc.out.path}*

# Runs standalone.
alone: jar
	${spark.root}/bin/spark-submit --class ${class.name} --master ${local.master} --name "${app.name}" ${jar.name} ${local.raw_input_path} ${local.preproc_out_path} ${local.training_path} ${local.testing_path} ${local.validation_path} ${local.output_path} ${local.model_in_path} ${local.model_out_path} ${local.metrics_file_path} ${local.max_depth} ${local.max_bins} ${local.num_trees} 4

# Runs standalone pre-proc.
alone-p:
	${spark.root}/bin/spark-submit --class PreProcess --master ${local.master} --name "${app.name}" ${jar.name} ${local.raw_input_path} ${local.preproc_out_path} ${local.training_path} ${local.testing_path} ${local.validation_path} ${local.output_path} ${local.model_in_path} ${local.model_out_path} ${local.metrics_file_path} ${local.max_depth} ${local.max_bins} ${local.num_trees} 4

# Runs standalone training.
alone-t: clean-local.model
	${spark.root}/bin/spark-submit --class ModelTraining --master ${local.master} --name "${app.name}" ${jar.name} ${local.raw_input_path} ${local.preproc_out_path} ${local.training_path} ${local.testing_path} ${local.validation_path} ${local.output_path} ${local.model_in_path} ${local.model_out_path} ${local.metrics_file_path} ${local.max_depth} ${local.max_bins} ${local.num_trees} 4

# Runs standalone classification.
alone-c: clean-local.output
	${spark.root}/bin/spark-submit --class FinalClassification --master ${local.master} --name "${app.name}" ${jar.name} ${local.raw_input_path} ${local.preproc_out_path} ${local.training_path} ${local.testing_path} ${local.validation_path} ${local.output_path} ${local.model_in_path} ${local.model_out_path} ${local.metrics_file_path} ${local.max_depth} ${local.max_bins} ${local.num_trees} 4

# Create S3 bucket.
make-bucket:
	aws s3 mb s3://${aws.bucket.name}

# Uploading data to aws
upload-conf-aws:
	aws s3 sync ${local.python.dir} s3://${aws.bucket.name}/${aws.python.dir}

# Upload raw input
upload-raw-input-aws:
	aws s3 sync ${local.raw_input_path} s3://${aws.bucket.name}/${aws.raw_input_path}
	
# Upload preproc input
upload-preproc-input-aws:
	aws s3 sync ${local.training_path} s3://${aws.bucket.name}/${aws.training_path}
	aws s3 sync ${local.testing_path} s3://${aws.bucket.name}/${aws.testing_path}
	aws s3 sync ${local.validation_path} s3://${aws.bucket.name}/${aws.validation_path}

# Upload model
upload-model-aws:
	aws s3 sync ${local.model_in_path} s3://${aws.bucket.name}/${aws.model_in_path}

# Upload application to S3 bucket.
upload-app-aws:
	aws s3 cp ${jar.name} s3://${aws.bucket.name}/${aws.proj.dir}/${jar.name}
	#aws s3 sync ${local.conf.dir} s3://${aws.bucket.name}/${aws.conf.dir}

# Do all the upload steps
upload-all-aws: upload-conf-aws upload-raw-input-aws upload-preproc-input-aws
	echo "Uploaded all"

#--configurations "file://${local.conf.file}" \
#--bootstrap-actions Path=s3://${aws.bucket.name}/${aws.bootstrap.file},Args=["${aws.bucket.name}","${aws.conf.file}"] \
#		--applications Name=Hadoop Name=Spark \
#		--steps Type=CUSTOM_JAR,Name="${app.name}",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster","--conf","spark.yarn.appMasterEnv.PYSPARK_PYTHON=python34","--conf","spark.executorEnv.PYSPARK_PYTHON=python34","--master","yarn","--class","${class.name}","s3://${aws.bucket.name}/${jar.name}","s3://${aws.bucket.name}/${aws.training_path}","s3://${aws.bucket.name}/${aws.output}","s3://${aws.bucket.name}/${aws.test}","s3://${aws.bucket.name}/${aws.model}","s3://${aws.bucket.name}/${aws.python_file_path}"] \
# Main EMR launch. jar
cloud-p: delete-pre-proc-job-output
	aws emr create-cluster \
		--name "${job.name}" \
		--release-label ${aws.release.label} \
		--instance-groups '[{"InstanceCount":${aws.num.workers},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		--applications Name=Hadoop Name=Spark \
		--steps Type=CUSTOM_JAR,Name="${app.name}",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster","--master","yarn","--class","PreProcessing","s3://${aws.bucket.name}/${aws.proj.dir}/${jar.name}","s3://${aws.bucket.name}/${aws.raw_input_path}","s3://${aws.bucket.name}/${aws.preproc_out_path}/${aux_job_id}","s3://${aws.bucket.name}/${aws.training_path}","s3://${aws.bucket.name}/${aws.testing_path}","s3://${aws.bucket.name}/${aws.validation_path}","s3://${aws.bucket.name}/${aws.output_path}","s3://${aws.bucket.name}/${aws.model_in_path}","s3://${aws.bucket.name}/${aws.model_out_path}","s3://${aws.bucket.name}/${aws.metrics_file_path}","${aws.max_depth}","${aws.max_bins}","${aws.num_trees}","${aws.partition.size}"] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--service-role EMR_DefaultRole \
		--ec2-attributes InstanceProfile=EMR_EC2_DefaultRole,SubnetId=${aws.subnet.id} \
		--region ${aws.region} \
		--enable-debugging \
		--auto-terminate

cloud-t: delete-model-aws
	aws emr create-cluster \
		--name "${job.name}" \
		--release-label ${aws.release.label} \
		--instance-groups '[{"InstanceCount":${aws.num.workers},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		--applications Name=Spark \
		--steps Type=CUSTOM_JAR,Name="${app.name}",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster","--master","yarn","--class","ModelTraining","s3://${aws.bucket.name}/${aws.proj.dir}/${jar.name}","s3://${aws.bucket.name}/${aws.raw_input_path}","s3://${aws.bucket.name}/${aws.preproc_out_path}","s3://${aws.bucket.name}/${aws.training_path}","s3://${aws.bucket.name}/${aws.testing_path}","s3://${aws.bucket.name}/${aws.validation_path}","s3://${aws.bucket.name}/${aws.output_path}","s3://${aws.bucket.name}/${aws.model_in_path}","s3://${aws.bucket.name}/${aws.model_out_path}","s3://${aws.bucket.name}/${aws.metrics_file_path}","${aws.max_depth}","${aws.max_bins}","${aws.num_trees}","${aws.partition.size}"] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--service-role EMR_DefaultRole \
		--ec2-attributes InstanceProfile=EMR_EC2_DefaultRole,SubnetId=${aws.subnet.id} \
		--region ${aws.region} \
		--enable-debugging \
		--auto-terminate

cloud-c: delete-classification-output upload-model-aws
	aws emr create-cluster \
		--name "${job.name}" \
		--release-label ${aws.release.label} \
		--instance-groups '[{"InstanceCount":${aws.num.workers},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		--applications Name=Spark \
		--steps Type=CUSTOM_JAR,Name="${app.name}",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster","--master","yarn","--class","FinalClassification","s3://${aws.bucket.name}/${aws.proj.dir}/${jar.name}","s3://${aws.bucket.name}/${aws.raw_input_path}","s3://${aws.bucket.name}/${aws.preproc_out_path}","s3://${aws.bucket.name}/${aws.training_path}","s3://${aws.bucket.name}/${aws.testing_path}","s3://${aws.bucket.name}/${aws.validation_path}","s3://${aws.bucket.name}/${aws.output_path}/${aux_job_id}","s3://${aws.bucket.name}/${aws.model_in_path}","s3://${aws.bucket.name}/${aws.model_out_path}","s3://${aws.bucket.name}/${aws.metrics_file_path}","${aws.max_depth}","${aws.max_bins}","${aws.num_trees}","${aws.partition.size}"] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--service-role EMR_DefaultRole \
		--ec2-attributes InstanceProfile=EMR_EC2_DefaultRole,SubnetId=${aws.subnet.id} \
		--region ${aws.region} \
		--enable-debugging \
		--auto-terminate

cloud:
	aws emr create-cluster \
		--name "${job.name}" \
		--release-label ${aws.release.label} \
		--instance-groups '[{"InstanceCount":${aws.num.workers},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		--applications Name=Spark \
		--steps Type=CUSTOM_JAR,Name="${app.name}",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster","--master","yarn","--class","${class.name}","s3://${aws.bucket.name}/${aws.proj.dir}/${jar.name}","s3://${aws.bucket.name}/${aws.raw_input_path}","s3://${aws.bucket.name}/${aws.preproc_out_path}","s3://${aws.bucket.name}/${aws.training_path}","s3://${aws.bucket.name}/${aws.testing_path}","s3://${aws.bucket.name}/${aws.validation_path}","s3://${aws.bucket.name}/${aws.output_path}","s3://${aws.bucket.name}/${aws.model_in_path}","s3://${aws.bucket.name}/${aws.model_out_path}","s3://${aws.bucket.name}/${aws.metrics_file_path}","${aws.max_depth}","${aws.max_bins}","${aws.num_trees}","${aws.partition.size}"] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--service-role EMR_DefaultRole \
		--ec2-attributes InstanceProfile=EMR_EC2_DefaultRole,SubnetId=${aws.subnet.id} \
		--region ${aws.region} \
		--enable-debugging \
		--auto-terminate

# Package for release.
distro:
	rm Project.tar.gz
	rm Project.zip
	rm -rf build
	mkdir -p build/deliv/Project/main/scala/project
	cp -r src/main/scala/project/* build/deliv/Project/main/scala/project
	cp pom.xml build/deliv/Project
	cp Makefile build/deliv/Project
	cp README.txt build/deliv/Project
	tar -czf Project.tar.gz -C build/deliv Project
	cd build/deliv && zip -rq ../../Project.zip Project

clean-local.output:
	rm -rf ${local.output_path}*

clean-local.preproc.out.path-aws:
	rm -rf ${local.preproc.out.path}*

clean-local.model.input.path:
	rm -rf ${local.model_in_path}*

clean-local.model:
	rm -rf ${local.model_out_path}*
	rm -rf ${local.metrics_file_path}*

clean-local-logs-aws:
	rm -rf ${local.log}*
			
download-model-aws: clean-local.model.input.path
	mkdir -p data/models/${model_name}
	aws s3 sync s3://${aws.bucket.name}/${aws.proj.dir}/data/models/${model.name} data/models/${model.name}

download-output-aws: clean-local.preproc.out.path-aws
	mkdir -p ${local.preproc.out.path}
	aws s3 sync s3://${aws.bucket.name}/${aws.output} ${local.preproc.out.path}

download-logs-aws: clean-local-logs-aws
	mkdir ${local.log}
	aws s3 sync s3://${aws.bucket.name}/${aws.log.dir} ${local.log}
	
# Delete S3 output dir.
delete-pre-proc-job-output:
	aws s3 rm s3://${aws.bucket.name}/ --recursive --exclude "*" --include "${aws.preproc_out_path}/${aux_job_id}*"

delete-classification-output:
	aws s3 rm s3://${aws.bucket.name}/ --recursive --exclude "*" --include "${aws.output_path}/${aux_job_id}*"
	
delete-logs-aws:
	aws s3 rm s3://${aws.bucket.name}/ --recursive --exclude "*" --include "${aws.log.dir}*"

delete-model-aws:
	aws s3 rm s3://${aws.bucket.name}/ --recursive --exclude "*" --include "${aws.proj.dir}/data/models/${model.name}*"
