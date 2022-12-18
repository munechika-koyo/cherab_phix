{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :private-members: _generate_rays

   {% block all_methods %}

   {% if all_methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in all_methods %}
   {%- if not item.startswith('_') or item in ['__call__', '_generate_rays'] %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
