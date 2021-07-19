package br.usp.ime.bioinfogen.web.providers;

import javax.servlet.ServletContext;

import org.springframework.context.support.AbstractApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.springframework.web.context.ConfigurableWebApplicationContext;
import org.springframework.web.context.support.XmlWebApplicationContext;

import br.com.caelum.vraptor.ComponentRegistry;
import br.com.caelum.vraptor.ioc.spring.SpringProvider;

public class ApplicationSpringProvider
    extends
        SpringProvider
{

    @Override
    protected void registerCustomComponents(
        final ComponentRegistry registry )
    {

    }

    @Override
    protected ConfigurableWebApplicationContext getParentApplicationContext(
        final ServletContext context )
    {
        final AbstractApplicationContext ctx = new ClassPathXmlApplicationContext(
            "classpath:spring/applicationContext.xml" );
        ctx.registerShutdownHook();

        final XmlWebApplicationContext wctx = new XmlWebApplicationContext();
        wctx.setParent( ctx );
        wctx.setConfigLocation( "" );
        wctx.setServletContext( context );

        return wctx;
    }
}