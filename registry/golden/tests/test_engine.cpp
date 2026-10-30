// FILEPATH: tests/test_engine.cpp
#include <gtest/gtest.h>

TEST(EngineTest, Initialization) {
    Engine engine;
    ASSERT_TRUE(engine.init());
}

TEST(EngineTest, Cleanup) {
    Engine engine;
    engine.init();
    ASSERT_NO_THROW(engine.cleanup());
}
