function dxdt = cstr(t,x,uex)

  global Tsamp
  
  % ZOH the control vector
  try
      u = uex(fix(t/Tsamp)+1);
  catch
      u = uex(end);
  end
  
  c_AO = 5.1;
  k10  =  1.287e12;
  k20  =  1.287e12;
  k30  =  9.043e09;
  E1   =  -9758.3;
  E2   =  -9758.3;
  E3   =  -8560.0;
  T    = 1.1419108442079495e02;
  k_1  = k10*exp(E1/(273.15 + T));
  k_2  = k20*exp(E2/(273.15 + T));
  k_3  = k30*exp(E3/(273.15 + T));
  TIMEUNITS_PER_HOUR = 3600.0;

  dx1 = (1/TIMEUNITS_PER_HOUR)*(u*(c_AO - x(1))-k_1*x(1)-k_3*x(1)*x(1));
  dx2 = (1/TIMEUNITS_PER_HOUR)*(-u*x(2)+k_1*x(1)-k_2*x(2));

  dxdt = [dx1; dx2];
  
end