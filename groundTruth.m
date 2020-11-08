function [x1next,x2next] = groundTruth(x1,x2,u)

    global Tsamp
    
    if size(x1,2) ~= 1 || size(x2,2) ~= 1
        error('Inputs x1 and x2 must be long columns')
    end
    
    num = size(x1,1);
    
    x1next = zeros(num,1);
    x2next = zeros(num,1);
    
    for i = 1:num
        
        [t,x] = ode45(@(t,x) em_CSTR(t,x,u), [0 Tsamp], [x1(i); x2(i)]);

        x1next(i) = x(end,1);
        x2next(i) = x(end,2);
        
        %disp('hello')
    end
    
end