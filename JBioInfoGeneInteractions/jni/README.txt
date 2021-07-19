Para configurar a geração de código nativo no eclipse, pode-se seguir o link:
http://omtlab.com/java-run-javah-from-eclipse/

---- Exemplo ----
Run/External Tools/External Tools Configurations
Criar um novo launcher de "Program"

2. Configuração

Location:
/usr/local/bin/jdk1.6.0_22/bin/javah

WorkDiretory:
${project_loc}/bin/

Arguments:
 -jni -verbose -d "${project_loc}${system_property:file.separator}jni" ${java_type_name}