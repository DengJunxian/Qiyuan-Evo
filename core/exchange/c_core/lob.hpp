#pragma once
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct RuleConfig {
  double commission_rate = 0.00025;
  double stamp_duty_rate = 0.0005;
  double min_price_tick = 0.01;
  int min_trade_unit = 1;
  int board_lot = 100;
  bool enforce_min_trade_unit = false;
  bool enforce_board_lot = false;
  bool allow_odd_lots = true;
  bool strict_queue_timestamps = false;
  std::string timestamp_precision = "microsecond";
};

struct Order {
  std::string order_id;
  std::string agent_id;
  double timestamp;
  std::string symbol;
  std::string side;
  std::string order_type;
  double price;
  double quantity;
  std::string status;
  double filled_qty;

  double remaining_qty() const { return quantity - filled_qty; }
  bool is_filled() const { return filled_qty >= quantity; }
};

struct Trade {
  std::string trade_id;
  double price;
  double quantity;
  std::string maker_id;
  std::string taker_id;
  std::string maker_agent_id;
  std::string taker_agent_id;
  double timestamp;
  double buyer_fee;
  double seller_fee;
  double seller_tax;
};

class LimitOrderBook {
public:
  explicit LimitOrderBook(std::string symbol, RuleConfig rule_config = RuleConfig());
  ~LimitOrderBook() = default;

  std::vector<Trade> add_order(Order &order);
  bool cancel_order(std::string order_id);

  double get_best_bid();
  double get_best_ask();
  std::map<std::string, std::vector<std::pair<double, double>>> get_depth(int levels);

  void clear();
  void set_rule_config(const RuleConfig &rule_config);
  RuleConfig get_rule_config() const;

private:
  std::string symbol;
  RuleConfig rule_config;

  using OrderPtr = std::shared_ptr<Order>;

  std::map<double, std::vector<OrderPtr>, std::greater<double>> bids;
  std::map<double, std::vector<OrderPtr>, std::less<double>> asks;
  std::unordered_map<std::string, OrderPtr> order_lookup;

  std::vector<Trade> match_order(OrderPtr order);
  std::vector<Trade> match_against_asks(OrderPtr buy_order);
  std::vector<Trade> match_against_bids(OrderPtr sell_order);
  Trade execute_trade(OrderPtr buy_order, OrderPtr sell_order, OrderPtr taker, OrderPtr maker);

  void add_to_book(OrderPtr order);
  void remove_order(OrderPtr order);
  double normalize_price(double price) const;
  double normalize_quantity(double quantity) const;
  double normalize_timestamp(double timestamp) const;
  bool should_rest(const OrderPtr &order) const;
};
