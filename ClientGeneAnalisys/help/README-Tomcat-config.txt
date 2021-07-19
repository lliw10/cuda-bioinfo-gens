apache-tomcat-7.0.34

BUG:
Reported on: http://tomcat.10.n6.nabble.com/Tomcat-7-0-23-startup-freezes-at-quot-INFO-Deploying-web-application-directory-quot-td1988652.html

Solved on: http://tomcat.10.n6.nabble.com/Hanging-on-startup-td4571254.html
 
Tomcat Freeze on startup. To solve, edit the file catalina.sh adding the 
following java_opts.

apache-tomcat-7.0.34/bin/catalina.sh

JAVA_OPTS="$JAVA_OPTS -Djava.security.egd=file:/dev/./urandom"
JAVA_OPTS="$JAVA_OPTS -Djava.library.path=/dir/native"