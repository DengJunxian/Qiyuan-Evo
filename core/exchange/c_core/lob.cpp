#include "lob.hpp"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <random>
#include <sstream>

std::string generate_uuid() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 15);
  static std::uniform_int_distribution<> dis2(8, 11);

  std::stringstream ss;
  ss << std::hex;
  for (int i = 0; i < 8; i++)
    ss << dis(gen);
  ss << "-";
  for (int i = 0; i < 4; i++)
    ss << dis(gen);
  ss << "-4";
  for (int i = 0; i < 3; i++)
    ss << dis(gen);
  ss << "-";
  ss << dis2(gen);
  for (int i = 0; i < 3; i++)
    ss << dis(gen);
  ss << "-";
  for (int i = 0; i < 12; i++)
    ss << dis(gen);
  return ss.str();
}

double current_time() { return static_cast<double>(std::time(nullptr)); }

LimitOrderBook::LimitOrderBook(std::string symbol, RuleConfig rule_config)
    : symbol(std::move(symbol)), rule_config(rule_config) {}

void LimitOrderBook::set_rule_config(const RuleConfig &new_rule_config) {
  rule_config = new_rule_config;
}

RuleConfig LimitOrderBook::get_rule_config() const { return rule_config; }

double LimitOrderBook::normalize_price(double price) const {
  if (rule_config.min_price_tick <= 0.0)
    return price;
  double steps = std::round(price / rule_config.min_price_tick);
  double quantized = steps * rule_config.min_price_tick;
  return std::round(quantized * 1000000.0) / 1000000.0;
}

double LimitOrderBook::normalize_quantity(double quantity) const {
  double normalized = std::max(0.0, std::floor(quantity));
  int trade_unit = std::max(1, rule_config.min_trade_unit);
  if (rule_config.enforce_min_trade_unit) {
    normalized = std::floor(normalized / trade_unit) * trade_unit;
  }
  if (rule_config.enforce_board_lot && normalized >= rule_config.board_lot) {
    normalized = std::floor(normalized / rule_config.board_lot) * rule_config.board_lot;
  }
  return normalized;
}

double LimitOrderBook::normalize_timestamp(double timestamp) const {
  if (rule_config.timestamp_precision == "second") {
    return std::round(timestamp);
  }
  if (rule_config.timestamp_precision == "millisecond") {
    return std::round(timestamp * 1000.0) / 1000.0;
  }
  if (rule_config.timestamp_precision == "microsecond") {
    return std::round(timestamp * 1000000.0) / 1000000.0;
  }
  return timestamp;
}

bool LimitOrderBook::should_rest(const OrderPtr &order) const {
  return order->order_type == "limit" || order->order_type == "post-only";
}

std::vector<Trade> LimitOrderBook::add_order(Order &order_ref) {
  auto order = std::make_shared<Order>(order_ref);
  std::vector<Trade> trades;

  if (order->symbol != this->symbol) {
    order->status = "rejected";
    order_ref.status = "rejected";
    return trades;
  }

  order->timestamp = normalize_timestamp(order->timestamp);
  order->price = normalize_price(order->price);
  order->quantity = normalize_quantity(order->quantity);
  if (order->quantity <= 0.0) {
    order->status = "rejected";
    order_ref.status = "rejected";
    return trades;
  }

  trades = match_order(order);

  if (!order->is_filled()) {
    if (should_rest(order) && order->status != "cancelled" && order->status != "rejected") {
      add_to_book(order);
      order->status = (order->filled_qty > 0.0) ? "partial" : "pending";
    } else if (order->filled_qty > 0.0) {
      order->status = "partial";
    } else if (order->status != "rejected") {
      order->status = "cancelled";
    }
  }

  order_ref.filled_qty = order->filled_qty;
  order_ref.status = order->status;
  order_ref.price = order->price;
  order_ref.quantity = order->quantity;
  order_ref.timestamp = order->timestamp;

  return trades;
}

std::vector<Trade> LimitOrderBook::match_order(OrderPtr incoming) {
  if (incoming->side == "buy") {
    return match_against_asks(incoming);
  }
  return match_against_bids(incoming);
}

std::vector<Trade> LimitOrderBook::match_against_asks(OrderPtr buy_order) {
  std::vector<Trade> trades;
  std::vector<OrderPtr> to_remove;

  for (auto &[price, order_list] : asks) {
    if (buy_order->remaining_qty() <= 0.0)
      break;
    if (buy_order->price < price)
      break;

    for (auto &ask_order : order_list) {
      if (buy_order->remaining_qty() <= 0.0)
        break;
      Trade trade = execute_trade(buy_order, ask_order, buy_order, ask_order);
      trades.push_back(trade);
      if (ask_order->is_filled()) {
        to_remove.push_back(ask_order);
      }
    }
  }

  for (auto &order : to_remove) {
    remove_order(order);
  }
  return trades;
}

std::vector<Trade> LimitOrderBook::match_against_bids(OrderPtr sell_order) {
  std::vector<Trade> trades;
  std::vector<OrderPtr> to_remove;

  for (auto &[price, order_list] : bids) {
    if (sell_order->remaining_qty() <= 0.0)
      break;
    if (sell_order->price > price)
      break;

    for (auto &bid_order : order_list) {
      if (sell_order->remaining_qty() <= 0.0)
        break;
      Trade trade = execute_trade(bid_order, sell_order, sell_order, bid_order);
      trades.push_back(trade);
      if (bid_order->is_filled()) {
        to_remove.push_back(bid_order);
      }
    }
  }

  for (auto &order : to_remove) {
    remove_order(order);
  }
  return trades;
}

Trade LimitOrderBook::execute_trade(OrderPtr buy_order, OrderPtr sell_order,
                                    OrderPtr taker, OrderPtr maker) {
  double exec_qty = std::min(buy_order->remaining_qty(), sell_order->remaining_qty());
  double exec_price = maker->price;

  buy_order->filled_qty += exec_qty;
  sell_order->filled_qty += exec_qty;

  auto update_status = [](OrderPtr order) {
    if (order->is_filled())
      order->status = "filled";
    else if (order->filled_qty > 0.0)
      order->status = "partial";
  };
  update_status(buy_order);
  update_status(sell_order);

  double notional = exec_price * exec_qty;
  double trade_timestamp = rule_config.strict_queue_timestamps
                               ? std::max(normalize_timestamp(buy_order->timestamp),
                                          normalize_timestamp(sell_order->timestamp))
                               : current_time();

  Trade trade;
  trade.trade_id = generate_uuid();
  trade.price = exec_price;
  trade.quantity = exec_qty;
  trade.maker_id = maker->order_id;
  trade.taker_id = taker->order_id;
  trade.maker_agent_id = maker->agent_id;
  trade.taker_agent_id = taker->agent_id;
  trade.timestamp = trade_timestamp;
  trade.buyer_fee = notional * rule_config.commission_rate;
  trade.seller_fee = notional * rule_config.commission_rate;
  trade.seller_tax = notional * rule_config.stamp_duty_rate;
  return trade;
}

void LimitOrderBook::add_to_book(OrderPtr order) {
  order_lookup[order->order_id] = order;
  if (order->side == "buy") {
    bids[order->price].push_back(order);
  } else {
    asks[order->price].push_back(order);
  }
}

void LimitOrderBook::remove_order(OrderPtr order) {
  if (order->side == "buy") {
    auto it = bids.find(order->price);
    if (it != bids.end()) {
      auto &vec = it->second;
      vec.erase(std::remove(vec.begin(), vec.end(), order), vec.end());
      if (vec.empty())
        bids.erase(it);
    }
  } else {
    auto it = asks.find(order->price);
    if (it != asks.end()) {
      auto &vec = it->second;
      vec.erase(std::remove(vec.begin(), vec.end(), order), vec.end());
      if (vec.empty())
        asks.erase(it);
    }
  }
  order_lookup.erase(order->order_id);
}

bool LimitOrderBook::cancel_order(std::string order_id) {
  auto it = order_lookup.find(order_id);
  if (it == order_lookup.end())
    return false;

  OrderPtr order = it->second;
  if (order->status == "filled")
    return false;

  remove_order(order);
  order->status = "cancelled";
  return true;
}

double LimitOrderBook::get_best_bid() {
  if (bids.empty())
    return 0.0;
  return bids.begin()->first;
}

double LimitOrderBook::get_best_ask() {
  if (asks.empty())
    return 0.0;
  return asks.begin()->first;
}

std::map<std::string, std::vector<std::pair<double, double>>> LimitOrderBook::get_depth(int levels) {
  std::map<std::string, std::vector<std::pair<double, double>>> result;

  int count = 0;
  for (auto &[price, orders] : bids) {
    if (count++ >= levels)
      break;
    double total_qty = 0.0;
    for (auto &order : orders)
      total_qty += order->remaining_qty();
    result["bids"].push_back({price, total_qty});
  }

  count = 0;
  for (auto &[price, orders] : asks) {
    if (count++ >= levels)
      break;
    double total_qty = 0.0;
    for (auto &order : orders)
      total_qty += order->remaining_qty();
    result["asks"].push_back({price, total_qty});
  }

  return result;
}

void LimitOrderBook::clear() {
  bids.clear();
  asks.clear();
  order_lookup.clear();
}
