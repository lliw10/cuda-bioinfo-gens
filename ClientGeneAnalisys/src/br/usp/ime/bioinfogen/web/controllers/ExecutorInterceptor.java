package br.usp.ime.bioinfogen.web.controllers;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.jsp.jstl.core.Config;

import br.com.caelum.vraptor.InterceptionException;
import br.com.caelum.vraptor.Intercepts;
import br.com.caelum.vraptor.core.InterceptorStack;
import br.com.caelum.vraptor.interceptor.Interceptor;
import br.com.caelum.vraptor.ioc.RequestScoped;
import br.com.caelum.vraptor.resource.ResourceMethod;

@Intercepts
@RequestScoped
public class ExecutorInterceptor
    implements
        Interceptor
{
    private final HttpServletRequest request;

    // private final Result result;
    // private final Localization loc;

    public ExecutorInterceptor(
        final HttpServletRequest request )
    {
        this.request = request;
    }

    // public LocaleInterceptor(
    // final HttpServletRequest request,
    // final Result result,
    // final Localization loc )
    // {
    // this.request = request;
    // this.result = result;
    // this.loc = loc;
    // }

    @Override
    public boolean accepts(
        final ResourceMethod m )
    {
        return true;
    }

    @Override
    public void intercept(
        final InterceptorStack stack,
        final ResourceMethod method,
        final Object resourceInstance )
        throws InterceptionException
    {
        if( request.getParameter( "idioma" ) != null ) {
            final String language = request.getParameter( "idioma" );
            // Locale locale = new Locale(language);
            Config.set( request.getSession(), Config.FMT_LOCALE, language );
            Config.set( request.getSession(), Config.FMT_FALLBACK_LOCALE, language );
        }
        System.out.println( "Interceptando " + request.getRequestURI() );

        stack.next( method, resourceInstance );

    }
}