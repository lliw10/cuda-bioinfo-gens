package br.usp.ime.bioinfogen.remote;

import org.springframework.beans.factory.FactoryBean;

public class BaseUrlFactoryBean
    implements
        FactoryBean
{
    private final String remoteServicesBaseUrl;
    private final String url;

    public BaseUrlFactoryBean(
        final String url,
        final String urlCompl )
    {
        this.url = url;
        this.remoteServicesBaseUrl = urlCompl;
    }

    @Override
    public Object getObject()
        throws Exception
    {
        return url + remoteServicesBaseUrl;
    }

    @Override
    public Class<?> getObjectType()
    {
        return String.class;
    }

    @Override
    public boolean isSingleton()
    {
        return true;
    }
}
