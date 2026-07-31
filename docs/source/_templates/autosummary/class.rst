{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :member-order: bysource
   :exclude-members: __init__, __new__

   .. raw:: html

      <hr class="api-member-divider">

   .. automethod:: __init__
{% if 'reset' in inherited_members %}

   .. automethod:: reset
{% endif %}
