def template(): 
    r'''

    One Line Description 

    Parameters
    ----------

    variable_name : type
        Description

    Returns
    -------

    variable_name : type
        Description 

    Examples
    --------
      >>> from astropy.table import Table, TableAttribute
      >>> class MyTable(Table):
      ...     identifier = TableAttribute(default=1)
      >>> t = MyTable(identifier=10)
      >>> t.identifier
      10
      >>> t.meta
      OrderedDict([('__attributes__', {'identifier': 10})])
      
    '''