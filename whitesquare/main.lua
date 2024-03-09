local x = 400
local y = 300

function love.draw()
    love.graphics.rectangle("fill", x, y, 10, 10)
end

function love.update(dt)
    if love.keyboard.isDown("right") then
        x = x + 1
    end
end
