# Elastic Meetup - Nginx Demo
This is the demo environment for https://www.meetup.com/Turkey-Elastic-Fantastics/events/244682578/ Elastic-Turkey meetup.

## Credits: 
https://github.com/xeraa/vagrant-elastic-stack -- 
This is a mini version of a more complete vagrant-elastic-stack as outlined here.

 https://github.com/splunk/eventgen/tree/develop/samples for samples in example data generation with some minor modifications to fit the code :)


## Features

* Filebeat modules for nginx and system (initally commented out)
* Sample nginx logs and a simple generator python file

## Vagrant and Ansible

Do a simple `vagrant up` by using [Vagrant](https://www.vagrantup.com)'s [Ansible provisioner](https://www.vagrantup.com/docs/provisioning/ansible.html). All you need is a working [Vagrant installation](https://www.vagrantup.com/docs/installation/) (1.8.6+ but the latest version is always recommended), a [provider](https://www.vagrantup.com/docs/providers/) (tested with the latest [VirtualBox](https://www.virtualbox.org) version), and 2.5GB of RAM.

With the [Ansible playbooks](https://docs.ansible.com/ansible/playbooks.html) in the */elastic-stack/* folder you can configure the whole system step by step. Just run them in the given order inside the Vagrant box:

NOTE: Initially you will not need logstash.  It is there for further testing/improvement purposes.

```
> vagrant ssh
$ ansible-playbook /elastic-stack/1_configure-elasticsearch.yml
$ ansible-playbook /elastic-stack/2_configure-kibana.yml
$ ansible-playbook /elastic-stack/4_configure-filebeat.yml
$ ansible-playbook /elastic-stack/5_configure-dashboards.yml
```

Or if you are in a hurry, run all playbooks with `$ /elastic-stack/all.sh` at once.

## Kibana

Access Kibana at [http://localhost:5601](http://localhost:5601)


## Nginx Example Data
You can run the given `nginxevents.py` code to generate events in `logs/example_access.log`. By default this is read by filebeat.
```
python3 nginxevents.py
```

